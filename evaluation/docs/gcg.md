# 🧩 GCG

[🗂️ Dataset guides](../README.md)

Evaluate Qwen3 models against **precomputed GCG suffixes**, with 100 examples per
input file. This package replays attacks; it does not search for new suffixes.

## 1. Setup

From the **CBC-PI repository root**, use the shared environment:

```bash
conda activate cbc_test
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
export CBC_MODELS_DIR=/absolute/path/to/models
cd evaluation/gcg
```

## 2. ▶️ Run

Check the CBC Qwen3 DPI inputs without loading a model:

```bash
python batch_generate.py \
  --input dpi/vllm_grpo_qwen3_0211.json \
  --model-path "${CBC_MODELS_DIR}/grpo_qwen3_0211" \
  --output outputs/dpi/cbc_qwen3.json --dry-run
```

Then run generation on your selected GPU:

```bash
CUDA_VISIBLE_DEVICES=2 python batch_generate.py \
  --input dpi/vllm_grpo_qwen3_0211.json \
  --model-path "${CBC_MODELS_DIR}/grpo_qwen3_0211" \
  --output outputs/dpi/cbc_qwen3.json
```

For the original CBC IPI configuration, use
`--input ipi/vllm_data_grpo_qwen3_0211.json --model-type vllm_data` and a new output
path. The original generator calls `batch_generate_with_system` for both settings;
this import preserves that interface.

| Parameter | Default / meaning |
| :--- | :--- |
| `--input`, `--output` | Required input and new output JSON paths. |
| `--model-path` | Required local checkpoint directory. |
| `--model-type` | `vllm`; also `vllm_data`, `sandwich`, `promptarmor`. |
| `--n-responses` | `3` generations per example. |
| `--temperature` | `0.8`; original top-p `0.9` and max output tokens `4096` remain in the wrappers. |
| `--tensor-parallel-size` | `1`; set `2` with `CUDA_VISIBLE_DEVICES=2,3` for two GPUs. |
| `--max-model-len` | `9000`. |
| `--gpu-memory-utilization` | `0.5`. |
| `--promptarmor-model` | Filter model override; otherwise `PROMPTARMOR_MODEL` or `kimi-k2.6`. |
| `--dry-run` | Validate inputs and paths without inference or API calls. |

Match the attack input file to the model/defense used to generate its suffixes.
Using another checkpoint evaluates attack transfer. PromptArmor additionally needs
`ARK_API_KEY` and `ARK_BASE_URL`, set in the shell or local `.env` (see
[`.env.example`](../gcg/.env.example)).

## 3. Outputs

Each output row contains `instruction`, `input`, `goal`, `best_string`, and an
`outputs` list. Existing files are protected from overwrite. Historical generation
results are excluded, and automatic ASR/DSR scoring is pending label validation.
