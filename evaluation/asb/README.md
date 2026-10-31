<div align="center">

# 🛡️ ASB Evaluation for CBC

**Configure a model · Run prompt-injection evaluation · Summarize results**

<a href="../README.md">🗂️ All benchmarks</a> ·
<a href="#tutorial">🚀 Tutorial</a> ·
<a href="#configuration">⚙️ Config reference</a> ·
<a href="#run">▶️ Run commands</a> ·
<a href="#results">📊 Results</a>

</div>

This package contains the author's ASB experiment code for direct prompt injection
and task utility evaluation. See [source and modifications](UPSTREAM.md),
[upstream documentation and citation](README.upstream.md), and the [MIT license](LICENSE).

> 🛠️ **Validation status:** environment installation, core imports, Python 3.11
> syntax checks, three CBC config dry runs, and four launcher/summary checks passed.
> Full model inference validation in the fresh environment is pending.

<a id="tutorial"></a>
## 🚀 Tutorial

Follow these steps from the **CBC-PI repository root**. After entering
`evaluation/asb`, run all remaining commands from that directory.

<a id="environment"></a>
### 1. 📦 Prepare the environment

```bash
conda create -n cbc_test python=3.11 pip -y
conda activate cbc_test
export PYTHONNOUSERSITE=1
cd evaluation/asb
python -m pip install -r requirements.txt
python -m pip check
```

If `cbc_test` is already installed, activate it and enter `evaluation/asb`.
`PYTHONNOUSERSITE=1` isolates the environment from packages in `~/.local`.

| Component | Version |
| :--- | :--- |
| Python | 3.11 |
| PyTorch | 2.7.0 |
| vLLM | 0.9.1 |
| Transformers | 4.53.0 |

[Root `requirements.txt`](../../requirements.txt) is the complete dependency export from the server’s `cbc_test`
environment (`PYTHONNOUSERSITE=1 python -m pip freeze --local`), with 207 packages
and their exact versions. It includes PyTorch, vLLM, and CUDA dependencies.
`evaluation/asb/requirements.txt` references that root export.
`requirements-cuda.txt` remains a compatible installation entry point;
`requirements-lock.txt` preserves the initial validated environment snapshot.
These versions describe the new validation environment; the exact historical
paper environment has not been established.

### 2. 📁 Set model paths and judge credentials

Download the checkpoints from the [project model links](../../README.md#model-downloads).
Set their parent directory in your shell:

```bash
export CBC_MODELS_DIR=/absolute/path/to/models
```

For example, the existing configs expect folders such as:

```text
models/
├── grpo_qwen_0202/        # CBC: Qwen2.5-7B-Instruct
├── grpo_llama3_0204/      # CBC: Llama-3.1-8B-Instruct
├── grpo_qwen3_0211/       # CBC: Qwen3-14B
└── grpo_qwen_2_1/         # Checkpoint selected by DPI_2_1.yml
```

Alternatively, copy [`.env.example`](.env.example) to `.env` if you do not already
have one, then fill in its values. The launcher loads `evaluation/asb/.env`;
exported shell variables take precedence. Keep credentials in your local `.env`.

| Variable | What to fill in | When needed |
| :--- | :--- | :--- |
| `CBC_MODELS_DIR` | Absolute parent directory of your checkpoint folders | Configs containing `${CBC_MODELS_DIR}` |
| `GLM_API_KEY` | API key for your OpenAI-compatible refusal judge | Default evaluation with refusal judging |
| `GLM_BASE_URL` | Base URL of that judge endpoint | Default evaluation with refusal judging |
| `ASB_JUDGE_MODEL` | Model identifier accepted by the judge endpoint; default `deepseek-ai/DeepSeek-V3.2` | Override when your endpoint uses another judge |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI model/embedding credentials and endpoint | OpenAI models or memory embeddings |
| `ARK_API_KEY`, `ARK_BASE_URL` | PromptArmor judge credentials and endpoint | PromptArmor baseline |

The refusal judge sends evaluated model responses to the configured endpoint.
To evaluate without that metric, explicitly use `--skip-refusal-judge`;
refusal results will be `N/A`.

<a id="configuration"></a>
### 3. ⚙️ Write the YAML config

A config selects **the injection setting, attacker tools, model(s), and attack
variant(s)**. Here is a complete example based on
[`config/qwen2/DPI_2_1.yml`](config/qwen2/DPI_2_1.yml), with optional fields shown
explicitly:

```yaml
# Evaluation setting
injection_method: direct_prompt_injection

# Attacker-tool subset: all, agg, non-agg, or test
attack_tool:
  - all

# Memory is disabled for the standard CBC direct-injection run
read_db: false
write_db: false

# Format: backend::checkpoint_path
llms:
  - vllm::${CBC_MODELS_DIR}/grpo_qwen_2_1

# Prompt-injection variant(s)
attack_types:
  - combined_attack

# Optional: null means no additional ASB prompt defense
# CBC behavior comes from the checkpoint above
defense_type: null

# Optional: appended to output filenames
suffix: ""
```

#### YAML field reference

| Field | Required? | How to fill it in |
| :--- | :--- | :--- |
| `injection_method` | Yes | Use `direct_prompt_injection` for the CBC DPI evaluation. Other launcher values are listed below. |
| `attack_tool` | Yes | Non-empty YAML list. `all`: all attacker tools; `agg`: aggressive only; `non-agg`: non-aggressive only; `test`: the bundled small tool subset. |
| `llms` | Yes | Non-empty list of model specifications. For local CBC checkpoints use `vllm::/absolute/path/to/checkpoint` or `vllm::${CBC_MODELS_DIR}/folder_name`. |
| `attack_types` | Yes | Non-empty list chosen from `combined_attack`, `context_ignoring`, `escape_characters`, `fake_completion`, `naive`. |
| `read_db` | No | Boolean; defaults to `false`. Set `true` only after preparing the required memory database. |
| `write_db` | No | Boolean; defaults to `false`. Memory runs require a prepared database directory. |
| `defense_type` | No | Omit or set `null` for CBC without an extra ASB prompt defense. Examples of upstream defense values: `delimiters_defense`, `instructional_prevention`, `direct_paraphrase_defense`, `ob_sandwich_defense`. Their applicable attack settings differ; see upstream documentation. |
| `suffix` | No | Filename suffix, e.g. `"seed0"`. Defaults to `""`. Use a filename component without `/` or `\`. |
| `tasks_path` | No | Task JSONL path; defaults to bundled `data/agent_task.jsonl`. A relative custom path is resolved from your shell's working directory. CLI `--tasks_path` takes precedence. |

The launcher runs every **attacker-tool subset × model × attack variant**
combination in sequence. For example, two models and two attack variants with
`attack_tool: [all]` produce four evaluation runs.

<details>
<summary>🔎 Other injection settings and model backends</summary>

| `injection_method` | Setting |
| :--- | :--- |
| `direct_prompt_injection` | User-prompt injection |
| `observation_prompt_injection` | Tool-observation injection |
| `clean` | No prompt injection |
| `mixed_attack`, `DPI_OPI` | Direct + observation injection |
| `memory_attack` | Memory attack |
| `DPI_MP` | Direct injection; enable `read_db` to use prepared memory |
| `OPI_MP` | Observation injection; enable `read_db` to use prepared memory |

The initial validation focuses on **direct prompt injection without memory**.
Memory databases are not bundled. Memory and PoT evaluation have not been validated
in this release.

| Model specification | Backend |
| :--- | :--- |
| `vllm::/path/to/checkpoint` | Local inference, including CBC checkpoints |
| `cot::/path/to/checkpoint` | CoT baseline backend |
| `sandwich::/path/to/checkpoint` | Sandwich baseline backend |
| `promptarmor::/path/to/checkpoint` | PromptArmor baseline backend |
| `ollama/llama3:8b` | Ollama backend; requires a running Ollama service |
| `gpt-4o-mini` | Registered API model; requires provider credentials |

Backend prefixes and `defense_type` serve different purposes: prefixes select a
model wrapper; `defense_type` selects an extra ASB prompt defense. CBC evaluation
uses the CBC checkpoint with the `vllm::` prefix.

</details>

#### Which existing config should I choose?

| Evaluation | Config | Checkpoint folder |
| :--- | :--- | :--- |
| CBC · Qwen2.5-7B | [`config/qwen2/DPI_RL.yml`](config/qwen2/DPI_RL.yml) | `grpo_qwen_0202` |
| CBC · Llama-3.1-8B | [`config/llama3/DPI_RL.yml`](config/llama3/DPI_RL.yml) | `grpo_llama3_0204` |
| CBC · Qwen3-14B | [`config/qwen3/DPI_RL.yml`](config/qwen3/DPI_RL.yml) | `grpo_qwen3_0211` |
| Qwen2.5 checkpoint variant `2_1` | [`config/qwen2/DPI_2_1.yml`](config/qwen2/DPI_2_1.yml) | `grpo_qwen_2_1` |

`DPI_Base.yml` selects the original backbone and `DPI_SFT.yml` selects the
configured cold-start checkpoint. Other baseline and ablation configs require
their corresponding weights; they have not all been validated.

<a id="run"></a>
### 4. ▶️ Run evaluation

First check your config and model path. This prints the planned case count and
command without loading weights or making API calls:

```bash
python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_2_1.yml \
  --output_dir logs/run_2 \
  --dry-run
```

Then run on GPUs **2 and 3**, using the same command structure as the original
experiments:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_2_1.yml \
  --output_dir logs/run_2
```

This example uses `grpo_qwen_2_1`. To evaluate the released CBC Qwen2.5 checkpoint,
change the config to `config/qwen2/DPI_RL.yml`:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_RL.yml \
  --output_dir logs/cbc_qwen2_run_1
```

Without refusal-judge credentials, add the explicit skip option:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_2_1.yml \
  --output_dir logs/run_2_no_judge \
  --skip-refusal-judge
```

To use another checkpoint without editing YAML:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_RL.yml \
  --model vllm::/absolute/path/to/checkpoint \
  --output_dir logs/custom_model_run_1 \
  --skip-refusal-judge
```

#### Command-line parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `--cfg_path PATH` | Required | YAML config to load. Relative to the shell's working directory. |
| `--output_dir DIR` | `logs` | Base directory for logs and per-sample CSVs. Relative to the shell's working directory. |
| `--model SPEC` | Config `llms` | Replace the entire configured model list with one model specification. |
| `--task_num N` | `1` | Maximum tasks **per agent**, not total examples. Each selected task is paired with matching attacker tools. |
| `--tasks_path PATH` | Config value or bundled task file | Override the task JSONL. |
| `--dry-run` | Off | Validate inputs and print planned commands; no model inference or API calls. |
| `--skip-refusal-judge` | Off | Skip the external refusal metric; write `N/A`. ASR and task utility are still computed. |
| `-h`, `--help` | — | Show launcher help. |

`CUDA_VISIBLE_DEVICES=2,3` exposes physical GPUs 2 and 3 to this process.
Choose GPUs with sufficient free memory. The standard vLLM backend uses two GPUs
by default.

| GPU environment variable | Default | Meaning |
| :--- | :--- | :--- |
| `ASB_TENSOR_PARALLEL_SIZE` | `2` | Number of visible GPUs used for tensor parallel inference. |
| `ASB_MAX_MODEL_LEN` | `16192` | Maximum model context length. |
| `ASB_GPU_MEMORY_UTILIZATION` | `0.5` for standard vLLM; `0.8` for CoT/Sandwich/PromptArmor backends | vLLM GPU-memory utilization setting. |

For example, to use one GPU, provided the model and context fit:

```bash
CUDA_VISIBLE_DEVICES=2 ASB_TENSOR_PARALLEL_SIZE=1 \
  python scripts/agent_attack.py \
  --cfg_path config/qwen2/DPI_RL.yml \
  --output_dir logs/cbc_qwen2_single_gpu \
  --skip-refusal-judge
```

The inherited vLLM settings are temperature `0.1`, top-p `0.95`, and up to `4096`
output tokens. The evaluator sets seed `0`. Changing `logs/run_1` to `logs/run_2`
changes the output location; it does **not** change the seed.

<a id="results"></a>
### 5. 📊 Read and summarize results

```bash
python scripts/summarize.py logs/run_2
```

Outputs follow this structure:

```text
logs/run_2/
└── direct_prompt_injection/
    └── <backend_model>/
        └── no_memory/
            ├── combined_attack-all_.csv
            └── combined_attack-all_.log
```

A non-empty `suffix` is appended after the underscore in each filename.
With a configured `defense_type`, its name replaces `no_memory` in the path;
`read_db: true` uses `new_memory` when no extra defense is selected.
Existing log or CSV paths are protected from overwrite: choose a new output
directory or suffix for another run.

| Summary field | Meaning |
| :--- | :--- |
| `cases` | Number of completed rows in the CSV. |
| `ASR` | Fraction of examples with attacker success. |
| `DSR` | `1 - ASR`. |
| `original_task_success_rate` | Fraction satisfying the expected normal-tool achievements, using the original ASB checks. |
| `refusal_rate` | Fraction judged as refusal; `null` when all rows explicitly skip judging. |

Rates are fractions in `[0, 1]`. An invalid judge response is stored as `ERROR`
and rejected by the summarizer. An interrupted run can leave a partial CSV;
compare `cases` with the launcher's planned count before reporting results.
The summary reports each file separately and does not aggregate repeated runs.

---

[⬆️ Back to tutorial](#tutorial) · [🗂️ All benchmarks](../README.md) · [🏠 Project home](../../README.md)

### PromptArmor filter model

Set `promptarmor_model: kimi-k2.6` in a config using `promptarmor::`, or override
with `--promptarmor-model MODEL_ID`. `PROMPTARMOR_MODEL` in `.env` is also
supported. Priority is CLI → YAML → environment → default `kimi-k2.6`.
The model must be served by `ARK_BASE_URL`; credentials use `ARK_API_KEY`.
This selects the injection filter, while `llms` selects the task checkpoint.
