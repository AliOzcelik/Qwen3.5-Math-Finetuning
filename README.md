# Qwen 3.5 2B SFT and Benchmarking

This repo contains a small supervised fine-tuning workflow for `Qwen/Qwen3.5-2B`, plus benchmark scripts to compare the raw base model against the merged finetuned model.

## Raw vs Merged Benchmark Comparison

| Benchmark | Qwen 3.5 2B | Merged SFT Model | Delta |
|-----------|------------------|------------------|-------|
| GSM8K | 0.66 | 0.74 | +0.08 |
| MATH-500 | 0.27 | 0.33 | +0.06 |
| Math-CoT-20k | 0.10 | 0.05 | -0.05 |
| ARC-Challenge | 0.21 | 0.29 | +0.08 |
| BoolQ | 0.75 | 0.74 | -0.01 |
| CommonsenseQA | 0.21 | 0.28 | +0.07 |
| WinoGrande | 0.52 | 0.51 | -0.01 |

## Dataset

100K sample from nvidia/OpenMathReasoning

## What is in this repo

- `finetune.ipynb`: dataset prep, QLoRA fine-tuning, adapter save, and merged-model export
- `preprocess_openmath.py`: preprocessing for `nvidia/OpenMathReasoning`
- `run_all_benchmarks.py`: script-based evaluation for raw vs merged comparison

## Training setup

- Base model: `Qwen/Qwen3.5-2B`
- Fine-tuning method: QLoRA 4-bit
- Adapter method: LoRA
- Merged output: `outputs/qwen3.5-2b-merged`

Main training data used in this workflow:

- `nvidia/OpenMathReasoning`
- `jasonrqh/Math-CoT-20k`
- additional instruction / domain datasets referenced in the repo preprocessing scripts

The focus is mainly math reasoning and instruction-following, so the evaluation is centered on reasoning-heavy benchmarks.

## Benchmarks used

Math benchmarks:

- `GSM8K`
- `MATH-500`
- `competition_math`
- `Math-CoT-20k`
- `MMLU math` subjects

Reasoning benchmarks:

- `BoolQ`
- `WinoGrande`
- `CommonsenseQA`
- `ARC-Challenge`

## Running evaluation

Run the full script comparison:

```bash
python run_all_benchmarks.py
```

This evaluates:

- raw model: `Qwen/Qwen3.5-2B`
- merged model: `outputs/qwen3.5-2b-merged`

## Results files

Generated outputs are written under `outputs/eval_results/`.

Important files:

- `outputs/eval_results/raw_vs_merged/`
- `outputs/eval_results/reasoning_raw_vs_merged/`
- `outputs/eval_results/script_runs/`

Useful result tables:

- per-prediction outputs
- summary accuracy by benchmark
- raw vs merged accuracy deltas
- pairwise example-level comparisons

## Notes

- Some datasets may fail to load if they are not cached locally or if Hugging Face access is unavailable.
- `competition_math` is handled with fallback dataset IDs because availability can vary.
- The merged model is the correct target for direct inference and benchmark comparison.
