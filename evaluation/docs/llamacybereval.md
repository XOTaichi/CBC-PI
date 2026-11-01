# 🔐 LlamaCyberEval

[🗂️ Dataset guides](../README.md)

Prompt-injection evaluation imported from the author's `LlamaCyberSecurity`
experiment directory, based on [PurpleLlama / CybersecurityBenchmarks](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks).

## 1. Setup

From the **CBC-PI repository root**:

```bash
conda activate cbc_test
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
export CBC_MODELS_DIR=/absolute/path/to/models
cd evaluation/llamacybereval
```

Configure `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and `JUDGE_MODEL` in the shell or local
`.env`; see [`.env.example`](../llamacybereval/.env.example).
The default judge identifier is `deepseek-v3.2`; use a model your endpoint serves.

## 2. ▶️ Run

```bash
CUDA_VISIBLE_DEVICES=2 python run.py \
  --model-path "${CBC_MODELS_DIR}/grpo_qwen3_0211" \
  --output-dir results/cbc_qwen3/single_run1
```

Add `--dry-run` to check inputs without loading a model or calling an API.
For the original multilingual subset, use a separate output directory and add:

```bash
--prompt-path datasets/prompt_injection/sample_250/prompt_injection_multilingual_machine_translated.json
```

| Option | Meaning |
| :--- | :--- |
| `--model-path` | Local CBC, backbone, or other defense checkpoint. |
| `--model-type` | `VLLM` (default), `SANDWICH`, or `PROMPTARMOR`. |
| `--judge-model` | Override `JUDGE_MODEL`. |
| `--only-generate` | Run real inference without an external judge; save responses only. |
| `--num-test-cases` | Default `0`: use the full selected file; a positive value uses upstream evenly spaced sampling. |
| `--output-dir` | New output directory; existing response/judge/stat files are protected. |

GPU settings: `CBC_TENSOR_PARALLEL_SIZE=1` and
`CBC_GPU_MEMORY_UTILIZATION=0.7` by default. For two GPUs, set
`CUDA_VISIBLE_DEVICES=2,3 CBC_TENSOR_PARALLEL_SIZE=2`.
PromptArmor also requires `ARK_API_KEY`, `ARK_BASE_URL`, and optionally
`PROMPTARMOR_MODEL` (default `kimi-k2.6`).

## 3. 📊 Results

```bash
python summarize.py results/cbc_qwen3/single_run1/judge.json
```

Pass both single-language and multilingual `judge.json` paths to combine them.
The summary reports case count, ASR and DSR as fractions; invalid judge responses
are rejected. Response text and category statistics are saved alongside the judgments.

The source files contain **251 single-language** and **250 sampled multilingual**
examples (501 combined); the full multilingual file contains 1,004 examples.
These source counts are preserved without claiming an exact 500-example paper split.