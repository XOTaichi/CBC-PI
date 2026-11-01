# 🕵️ InjecAgent

See the [usage guide](../docs/injecagent.md).

Imported from the author's `pi_bench/InjecAgent` experiment directory, based on
[uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent).
The original [license](LICENCE), [README](README.upstream.md), prompt templates,
base/enhanced datasets and tool-response cache are retained.

Changes: removed the hardcoded simulator credential and endpoint, deferred optional
provider/guardrail imports, parameterized GPU and PromptArmor model settings, and
added data-only dry runs. Scoring no longer initializes a model or API client.
Sample failures prevent a partial run from being reported as a complete score.
Historical results and checkout-specific batch scripts are excluded.

The prompted VLLM path is the initial release target. Fine-tuned-agent and optional
DataSentinel/TogetherAI adapters retain upstream behavior and need their external
dependencies; they have not been validated here. Real generation, parsing and scoring passed
for one DH and one DS first-step example with CBC Qwen2.5-7B; full evaluation is
pending. See [runtime checks](../docs/smoke-tests.md).
