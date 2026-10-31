# ASB

[🗂️ Dataset guides](../README.md)

Direct prompt-injection evaluation with ASR/DSR and task utility under attack.

## 1. Setup

Run from the **CBC-PI repository root**:

```bash
conda create -n cbc_test python=3.11 pip -y
conda activate cbc_test
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
python -m pip check
export CBC_MODELS_DIR=/absolute/path/to/models
cd evaluation/asb
```

The root [`requirements.txt`](../../requirements.txt) contains the complete dependency export from `cbc_test`, including PyTorch, vLLM, and Transformers. If the environment already exists, skip the creation command.

Download the [CBC checkpoints](../../README.md#model-downloads) into this directory, e.g. `$CBC_MODELS_DIR/grpo_qwen_0202`.

## 2. Demo

All commands below run from `evaluation/asb`. Check the configuration first:

```bash
python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_RL.yml \
  --output_dir logs/cbc_qwen2_demo \
  --dry-run
```

Run CBC Qwen2.5 on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_RL.yml \
  --output_dir logs/cbc_qwen2_demo
```

For other CBC models, use [`config/llama3/DPI_RL.yml`](../asb/config/llama3/DPI_RL.yml) or [`config/qwen3/DPI_RL.yml`](../asb/config/qwen3/DPI_RL.yml). Choose a new output directory for each run.

The demo skips external refusal judging. To include refusal rate, configure the judge credentials in [`.env.example`](../asb/.env.example) and remove `--skip-refusal-judge`.

## 3. 🛠️ Customize your own defense

Create `config/custom_defense.yml` under `evaluation/asb`:

```yaml
injection_method: direct_prompt_injection
attack_tool:
  - all  # all, agg, non-agg, or test
write_db: false
llms:
  - vllm::${CBC_MODELS_DIR}/your_defense_checkpoint
attack_types:
  - combined_attack
```

- **`injection_method`**: evaluation setting; use `direct_prompt_injection` for DPI.
- **`attack_tool`**: all, aggressive, non-aggressive, or test attacker tools.
- **`write_db`**: keep `false` for evaluation without memory writes.
- **`llms`**: `backend::checkpoint_path`; replace the folder with your own model. Absolute paths also work.
- **`attack_types`**: `combined_attack`, `context_ignoring`, `fake_completion`, `escape_characters`, or `naive`.

### 🤖 Backend and GPU settings

| Model entry | Backend |
| :--- | :--- |
| `vllm::/path/to/model` | Local defense checkpoint |
| `cot::/path/to/model` | CoT wrapper |
| `sandwich::/path/to/model` | Sandwich wrapper |
| `promptarmor::/path/to/model` | PromptArmor wrapper; requires `ARK_API_KEY` and `ARK_BASE_URL` |
| `ollama/llama3:8b` | Running Ollama service |
| `gpt-4o-mini` | Registered API model with provider credentials |

**PromptArmor filter model:** add this field to a config using `promptarmor::`:

```yaml
promptarmor_model: kimi-k2.6  # Replace with a model ID served by ARK_BASE_URL
```

Or set `PROMPTARMOR_MODEL` in `.env`, or pass `--promptarmor-model MODEL_ID`.
Priority: CLI → YAML → environment → default `kimi-k2.6`. The `llms` checkpoint
still selects the task model; this option selects the injection filter.

For a defense implemented in code, add its wrapper in
[`llm_classes/`](../asb/aios/llm_core/llm_classes/), register it in
[`llms.py`](../asb/aios/llm_core/llms.py), and allow its prefix in
[`agent_attack.py`](../asb/scripts/agent_attack.py).

vLLM settings use **environment variables**, not YAML fields:

| Variable | Default |
| :--- | :--- |
| `ASB_TENSOR_PARALLEL_SIZE` | `2` GPUs |
| `ASB_MAX_MODEL_LEN` | `16192` tokens |
| `ASB_GPU_MEMORY_UTILIZATION` | `0.5` for `vllm`; `0.8` for the other vLLM wrappers |

### ▶️ Run

```bash
CUDA_VISIBLE_DEVICES=2,3 ASB_TENSOR_PARALLEL_SIZE=2 \
python scripts/agent_attack.py \
  --cfg_path config/custom_defense.yml \
  --output_dir logs/my_defense_run_1
```

## 4. Results

```bash
python scripts/summarize.py logs/cbc_qwen2_demo
```

Reports include **ASR**, **DSR = 1 − ASR**, and **original task success rate**. Refusal rate is unavailable when the judge is skipped.

**Validation:** setup and launcher checks passed; full inference validation in the fresh environment is pending.

[Source and modifications](../asb/UPSTREAM.md) · [Upstream documentation](../asb/README.upstream.md)
