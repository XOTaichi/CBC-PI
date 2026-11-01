# 🕵️ InjecAgent

[🗂️ Dataset guides](../README.md)

Indirect prompt-injection evaluation based on [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent).
The CBC setup uses the **enhanced** setting: 510 direct-harm (DH) and 544
data-stealing (DS) examples, 1,054 in total.

## 1. Setup

From the **CBC-PI repository root**:

```bash
conda activate cbc_test
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
export CBC_MODELS_DIR=/absolute/path/to/models
cd evaluation/injecagent
```

Most simulated attacker-tool responses are bundled. A missing cached response
requires `INJECAGENT_SIMULATOR_API_KEY`, `INJECAGENT_SIMULATOR_BASE_URL`, and
optionally `INJECAGENT_SIMULATOR_MODEL` (default `DeepSeek-V3.2-Exp`).
The client also accepts `OPENAI_API_KEY` / `OPENAI_BASE_URL` as fallbacks.
See [`.env.example`](../injecagent/.env.example); no credentials are bundled.

## 2. ▶️ Run

```bash
CUDA_VISIBLE_DEVICES=2 python src/evaluate_prompted_agent.py \
  --model_type VLLM \
  --model_name "${CBC_MODELS_DIR}/grpo_qwen3_0211" \
  --setting enhanced --prompt_type InjecAgent \
  --output_dir results/cbc_qwen3/run1 \
  --use_cache
```

Add `--dry-run` to validate data without loading a model or calling an API.
Use another CBC checkpoint in `--model_name` for Qwen2.5 or Llama-3.1.

| Option | Meaning |
| :--- | :--- |
| `--model_type` | `VLLM` for local checkpoints. Additional upstream adapters are retained but not all validated. |
| `--model_name` | Checkpoint path; API adapters use a provider model ID. |
| `--setting` | `enhanced` for the CBC setup; `base` is also bundled. |
| `--prompt_type` | `InjecAgent` or `hwchase17_react`. |
| `--output_dir` | Output parent directory; choose a new directory for a new experiment. |
| `--use_cache` | Reuse generated responses from an interrupted run. |
| `--max-cases N` | Limit to N examples per attack type for a quick check; default `0` uses all. |
| `--only_first_step` | Skip the second DS step; changes the evaluation protocol. |
| `--pe sandwich` | Add the Sandwich wrapper. |
| `--guardrail promptarmor` | Add PromptArmor; configure `ARK_API_KEY`, `ARK_BASE_URL` and optional `PROMPTARMOR_MODEL`. |

GPU settings: `CBC_TENSOR_PARALLEL_SIZE=1` and
`CBC_GPU_MEMORY_UTILIZATION=0.7` by default. For two GPUs, set
`CUDA_VISIBLE_DEVICES=2,3 CBC_TENSOR_PARALLEL_SIZE=2`.

## 3. 📊 Results

```bash
python src/eval_only.py results/cbc_qwen3 --all
```

JSONL results are saved under
`<output_dir>/prompted_<backend-or-defense>_<checkpoint>_<prompt_type>/`.
The upstream scorer reports ASR and validity; the summary adds
`DSR = 100 - ASR-all (Total)` in percent. Compare completed row counts with the
expected 1,054 cases before reporting scores. Failed cases cause the run to exit
with an error rather than reporting a partial score.
