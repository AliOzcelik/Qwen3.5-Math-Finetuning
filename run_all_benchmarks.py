import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


RAW_MODEL_ID = "Qwen/Qwen3.5-2B"
MERGED_MODEL_ID = "outputs/qwen3.5-2b-merged"
RESULTS_DIR = "outputs/eval_results/script_runs"

SAMPLES_PER_BENCHMARK = 100
MATH_MAX_NEW_TOKENS = 512
REASONING_MAX_NEW_TOKENS = 256
SEED = 42


random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


def strip_latex(text):
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace(",", "")
    return text.strip()


def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def extract_final_answer_line(text):
    matches = re.findall(r"Final answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else None


def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    return matches[-1] if matches else None


def canonicalize_answer(text):
    if text is None:
        return ""
    text = strip_latex(text)
    text = text.replace(" ", "").replace("%", "")
    if text.endswith("."):
        text = text[:-1]
    return text


def maybe_to_float(text):
    text = canonicalize_answer(text)
    if not text:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    except Exception:
        return None


def answers_match(pred, gold, tol = 1e-6):
    pred_c = canonicalize_answer(pred)
    gold_c = canonicalize_answer(gold)
    if pred_c == gold_c and pred_c != "":
        return True
    pred_f = maybe_to_float(pred_c)
    gold_f = maybe_to_float(gold_c)
    if pred_f is not None and gold_f is not None:
        return math.isclose(pred_f, gold_f, rel_tol=tol, abs_tol=tol)
    return False


def build_messages(question):
    system = (
        "You are a careful reasoner. Solve the task step by step when needed. "
        "End with a line exactly like: Final answer: <answer>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def load_inference_model(model_id_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, dtype


def unload_model(model=None, tokenizer=None):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()


def generate_answer(model, tokenizer, question, max_new_tokens):
    messages = build_messages(question)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_from_candidates(candidates, max_samples):
    errors = []
    for candidate in candidates:
        try:
            if candidate.get("config") is None:
                ds = load_dataset(candidate["path"], split=candidate["split"])
            else:
                ds = load_dataset(candidate["path"], candidate["config"], split=candidate["split"])
            ds = ds.select(range(min(max_samples, len(ds))))
            return ds, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("All candidates failed:\n" + "\n".join(errors))


def extract_gold_answer(example, spec):
    raw = str(example[spec["answer_field"]])
    mode = spec["gold_extractor"]
    if mode == "gsm8k":
        match = re.findall(r"####\s*(.+)", raw)
        return match[-1].strip() if match else raw.strip()
    if mode == "boxed_or_last":
        return extract_boxed(raw) or extract_final_answer_line(raw) or extract_last_number(raw) or raw.strip()
    return raw.strip()


def extract_pred_answer(text, spec_type):
    if spec_type == "boolq":
        candidate = extract_final_answer_line(text) or text
        low = candidate.lower()
        if "yes" in low:
            return "yes"
        if "no" in low:
            return "no"
        return candidate.strip().lower()
    if spec_type in {"mcq", "mmlu_math", "winogrande"}:
        candidate = extract_final_answer_line(text) or text
        match = re.findall(r"\b([A-J])\b", candidate.upper())
        return match[-1] if match else candidate.strip()
    return extract_final_answer_line(text) or extract_boxed(text) or extract_last_number(text) or text.strip()


def answer_matches_for_spec(pred, gold, spec_type):
    if spec_type == "boolq":
        return canonicalize_answer(pred).lower() == canonicalize_answer(gold).lower()
    if spec_type in {"mcq", "mmlu_math", "winogrande"}:
        return canonicalize_answer(pred).upper() == canonicalize_answer(gold).upper()
    return answers_match(pred, gold)


def build_mcq_question(question, labels, texts, instruction):
    options = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return instruction + "\n\n" + question + "\n\n" + "\n".join(options)


def build_mmlu_question(row):
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    options = [f"{letters[i]}. {choice}" for i, choice in enumerate(row["choices"])]
    return (
        "Answer the multiple-choice math question. End with a line exactly like: Final answer: <letter>.\n\n"
        f"{row['question']}\n\n" + "\n".join(options)
    )


def prepare_examples_for_spec(spec, max_samples):
    if spec["type"] == "mmlu_math":
        subjects = spec["subjects"]
        per_subject = max(1, math.ceil(max_samples / len(subjects)))
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        examples = []
        sources = []
        for subject in subjects:
            loaded = False
            for path in spec["candidate_paths"]:
                try:
                    ds = load_dataset(path, subject, split=spec["split"])
                    ds = ds.select(range(min(per_subject, len(ds))))
                    for row in ds:
                        answer = row["answer"]
                        gold = letters[answer] if isinstance(answer, int) else str(answer).strip()
                        examples.append(
                            {
                                "question": build_mmlu_question(row),
                                "gold": gold,
                                "subject": subject,
                            }
                        )
                    sources.append({"path": path, "subject": subject, "split": spec["split"]})
                    loaded = True
                    break
                except Exception:
                    pass
            if not loaded:
                raise RuntimeError(f"Failed to load MMLU subject: {subject}")
        return examples[:max_samples], sources

    ds, source = load_from_candidates(spec["candidates"], max_samples)
    examples = []
    for row in ds:
        if spec["type"] == "qa":
            examples.append(
                {
                    "question": str(row[spec["question_field"]]),
                    "gold": extract_gold_answer(row, spec),
                }
            )
        elif spec["type"] == "boolq":
            question = (
                "Answer the question using the passage. End with a line exactly like: Final answer: yes or Final answer: no.\n\n"
                f"Passage: {row['passage']}\n\nQuestion: {row['question']}"
            )
            gold = "yes" if bool(row["answer"]) else "no"
            examples.append({"question": question, "gold": gold})
        elif spec["type"] == "winogrande":
            question = (
                "Choose the best option to fill the blank. End with a line exactly like: Final answer: A or Final answer: B.\n\n"
                f"{row['sentence']}\n\nA. {row['option1']}\nB. {row['option2']}"
            )
            gold = "A" if str(row["answer"]).strip() == "1" else "B"
            examples.append({"question": question, "gold": gold})
        elif spec["type"] == "mcq":
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            question = build_mcq_question(
                row["question"],
                labels,
                texts,
                "Choose the correct option. End with a line exactly like: Final answer: <option letter>.",
            )
            gold = str(row.get("answerKey", row.get("answer", ""))).strip()
            examples.append({"question": question, "gold": gold})
        else:
            raise ValueError(f"Unsupported type: {spec['type']}")
    return examples, source


MATH_BENCHMARKS = [
    {
        "name": "gsm8k",
        "type": "qa",
        "candidates": [{"path": "openai/gsm8k", "config": "main", "split": "test"}],
        "question_field": "question",
        "answer_field": "answer",
        "gold_extractor": "gsm8k",
        "max_new_tokens": MATH_MAX_NEW_TOKENS,
    },
    {
        "name": "math_500",
        "type": "qa",
        "candidates": [{"path": "HuggingFaceH4/MATH-500", "config": None, "split": "test"}],
        "question_field": "problem",
        "answer_field": "answer",
        "gold_extractor": "boxed_or_last",
        "max_new_tokens": MATH_MAX_NEW_TOKENS,
    },
    {
        "name": "competition_math",
        "type": "qa",
        "candidates": [
            {"path": "competition_math", "config": None, "split": "test"},
            {"path": "hendrycks/competition_math", "config": None, "split": "test"},
            {"path": "competition_math", "config": "default", "split": "test"},
        ],
        "question_field": "problem",
        "answer_field": "solution",
        "gold_extractor": "boxed_or_last",
        "max_new_tokens": MATH_MAX_NEW_TOKENS,
    },
    {
        "name": "math_cot_20k",
        "type": "qa",
        "candidates": [{"path": "jasonrqh/Math-CoT-20k", "config": None, "split": "train"}],
        "question_field": "question",
        "answer_field": "response",
        "gold_extractor": "boxed_or_last",
        "max_new_tokens": MATH_MAX_NEW_TOKENS,
    },
    {
        "name": "mmlu_math",
        "type": "mmlu_math",
        "subjects": [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "high_school_statistics",
        ],
        "candidate_paths": ["cais/mmlu"],
        "split": "test",
        "max_new_tokens": MATH_MAX_NEW_TOKENS,
    },
]


REASONING_BENCHMARKS = [
    {
        "name": "boolq",
        "type": "boolq",
        "candidates": [{"path": "google/boolq", "config": None, "split": "validation"}],
        "max_new_tokens": REASONING_MAX_NEW_TOKENS,
    },
    {
        "name": "winogrande",
        "type": "winogrande",
        "candidates": [{"path": "allenai/winogrande", "config": "winogrande_debiased", "split": "validation"}],
        "max_new_tokens": REASONING_MAX_NEW_TOKENS,
    },
    {
        "name": "commonsense_qa",
        "type": "mcq",
        "candidates": [{"path": "tau/commonsense_qa", "config": None, "split": "validation"}],
        "max_new_tokens": REASONING_MAX_NEW_TOKENS,
    },
    {
        "name": "arc_challenge",
        "type": "mcq",
        "candidates": [{"path": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "validation"}],
        "max_new_tokens": REASONING_MAX_NEW_TOKENS,
    },
]


ALL_BENCHMARKS = MATH_BENCHMARKS + REASONING_BENCHMARKS


def load_benchmark_examples():
    example_map = {}
    load_status = []
    for spec in ALL_BENCHMARKS:
        try:
            examples, source = prepare_examples_for_spec(spec, SAMPLES_PER_BENCHMARK)
            example_map[spec["name"]] = {"spec": spec, "source": source, "examples": examples}
            load_status.append(
                {
                    "benchmark": spec["name"],
                    "samples": len(examples),
                    "source": json.dumps(source),
                    "status": "ok",
                }
            )
        except Exception as exc:
            load_status.append(
                {
                    "benchmark": spec["name"],
                    "samples": 0,
                    "source": None,
                    "status": f"load_failed: {exc}",
                }
            )
    return example_map, pd.DataFrame(load_status)


def run_model_on_examples(model_label, model_id_or_path, example_map):
    rows = []
    model, tokenizer, dtype = load_inference_model(model_id_or_path)
    print(f"Loaded {model_label}: {model_id_or_path}")
    print(f"Dtype: {dtype}")
    try:
        for benchmark_name, payload in example_map.items():
            spec = payload["spec"]
            for idx, example in enumerate(tqdm(payload["examples"], desc=f"{model_label} - {benchmark_name}")):
                pred_text = generate_answer(model, tokenizer, example["question"], spec["max_new_tokens"])
                pred = extract_pred_answer(pred_text, spec["type"])
                row = {
                    "model_label": model_label,
                    "model_id": model_id_or_path,
                    "benchmark": benchmark_name,
                    "index": idx,
                    "question": example["question"],
                    "gold": example["gold"],
                    "prediction": pred,
                    "prediction_text": pred_text,
                    "correct": answer_matches_for_spec(pred, example["gold"], spec["type"]),
                    "dataset_source": json.dumps(payload["source"]),
                }
                if "subject" in example:
                    row["subject"] = example["subject"]
                rows.append(row)
    finally:
        unload_model(model, tokenizer)
    return rows


def save_outputs(compare_df: pd.DataFrame):
    compare_path = os.path.join(RESULTS_DIR, "all_benchmarks_predictions.csv")
    summary_path = os.path.join(RESULTS_DIR, "all_benchmarks_summary.csv")
    pivot_path = os.path.join(RESULTS_DIR, "all_benchmarks_accuracy_pivot.csv")
    pairwise_path = os.path.join(RESULTS_DIR, "all_benchmarks_pairwise.csv")

    compare_df.to_csv(compare_path, index=False)

    summary_df = (
        compare_df.groupby(["benchmark", "model_label"], as_index=False)
        .agg(samples=("correct", "size"), correct=("correct", "sum"))
    )
    summary_df["accuracy"] = summary_df["correct"] / summary_df["samples"]
    summary_df.to_csv(summary_path, index=False)

    pivot_df = summary_df.pivot(index="benchmark", columns="model_label", values="accuracy").reset_index()
    if "merged_qwen" in pivot_df.columns and "raw_qwen" in pivot_df.columns:
        pivot_df["delta_merged_minus_raw"] = pivot_df["merged_qwen"] - pivot_df["raw_qwen"]
    pivot_df.to_csv(pivot_path, index=False)

    pairwise_index = [c for c in ["benchmark", "subject", "index", "question", "gold"] if c in compare_df.columns]
    pairwise_df = compare_df.pivot(index=pairwise_index, columns="model_label", values=["prediction", "correct"]).reset_index()
    pairwise_df.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).rstrip("_") if isinstance(col, tuple) else col
        for col in pairwise_df.columns
    ]
    if "correct_raw_qwen" in pairwise_df.columns and "correct_merged_qwen" in pairwise_df.columns:
        pairwise_df["changed"] = pairwise_df["correct_raw_qwen"] != pairwise_df["correct_merged_qwen"]
        pairwise_df["improved"] = (~pairwise_df["correct_raw_qwen"]) & (pairwise_df["correct_merged_qwen"])
        pairwise_df["regressed"] = pairwise_df["correct_raw_qwen"] & (~pairwise_df["correct_merged_qwen"])
    pairwise_df.to_csv(pairwise_path, index=False)

    if "subject" in compare_df.columns and "mmlu_math" in set(compare_df["benchmark"]):
        subject_df = (
            compare_df[compare_df["benchmark"] == "mmlu_math"]
            .groupby(["subject", "model_label"], as_index=False)
            .agg(samples=("correct", "size"), correct=("correct", "sum"))
        )
        if len(subject_df) > 0:
            subject_df["accuracy"] = subject_df["correct"] / subject_df["samples"]
            subject_df.to_csv(os.path.join(RESULTS_DIR, "mmlu_subject_summary.csv"), index=False)

    return summary_df, pivot_df


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    example_map, load_df = load_benchmark_examples()
    load_df.to_csv(os.path.join(RESULTS_DIR, "benchmark_load_status.csv"), index=False)
    print(load_df)

    raw_rows = run_model_on_examples("raw_qwen", RAW_MODEL_ID, example_map)
    merged_rows = run_model_on_examples("merged_qwen", MERGED_MODEL_ID, example_map)

    compare_df = pd.DataFrame(raw_rows + merged_rows)
    summary_df, pivot_df = save_outputs(compare_df)

    print("\nSummary:")
    print(summary_df)
    print("\nPivot:")
    print(pivot_df)
    print(f"\nSaved results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
