# 🧩 GCG

See the [GCG guide](../docs/gcg.md) for setup and commands.

Imported from the author's `pi_bench/gcg_qwen3` experiment directory. This package
replays **frozen GCG attack suffixes** against Qwen3 models; it does not run the
GCG optimization/search process. `dpi/` and `ipi/` each contain seven model-specific
input files with 100 examples per file. The inputs retain their original fields,
including search-time `output` and `isinject`; new generations go to separate files.

The model wrappers and prompt construction are retained. Release preparation adds
portable CLI paths, configurable GPU allocation and PromptArmor filter model,
input/output validation, and overwrite protection. An unused external
DataSentinel path injection was removed.

Historical `outputs/`, `outputs copy/`, and caches are excluded. The scorer found
under the historical outputs is not bundled: its embedded API credential must not
be published, and its DPI-specific label matching needs separate validation.
New outputs contain generated responses, not recomputed ASR/DSR scores.

**Validation:** input schemas, CLI dry runs and CPU checks are validated separately
from full model inference. See the guide for current status.
