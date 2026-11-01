# 🔐 LlamaCyberEval

See the [usage guide](../docs/llamacybereval.md).

Imported from the author's `pi_bench/LlamaCyberSecurity` experiment directory.
The prompt-injection code derives from
[PurpleLlama / CybersecurityBenchmarks](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks);
its Meta copyright headers and [MIT license](LICENSE) are retained.

This release contains the prompt-injection benchmark, VLLM/OpenAI adapters,
existing prompt-defense wrappers, and source prompt-injection datasets.
Unrelated cybersecurity tasks, external DataSentinel modules, original scripts
with checkout-specific paths, and historical results are excluded.

Changes: standalone paths, lazy imports, environment-based GPU/filter/judge
configuration, checked outputs, dry runs, and strict Yes/No result summaries.
Thread failures propagate instead of silently producing partial benchmark scores.
The 251-row single-language file and 250-row multilingual subset are preserved.
One real CBC Qwen2.5-7B generation passed; external judging and full evaluation
remain untested. See [runtime checks](../docs/smoke-tests.md).
