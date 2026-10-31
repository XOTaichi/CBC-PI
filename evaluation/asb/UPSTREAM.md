# Source and local changes

Vendored from https://github.com/agiresearch/ASB at commit
`eac7bcf38c116f42b46e6d480e56e599b99e73c2`, including the CBC experiment
modifications from the author's server. The upstream MIT license is in LICENSE.
README.upstream.md preserves the source README; images referenced there are not bundled.

CBC modifications already present in the experiment copy include vLLM backends,
answer extraction, output formatting, defense baselines, model configs, and refusal judging.
Release preparation adds dependency pins, portable model paths, configurable GPU
allocation, a checked subprocess launcher, configuration validation, and CSV summaries.
Logs, model weights, credentials, caches and prepopulated memory databases are excluded.
The main release path is direct prompt injection with no memory; upstream memory/PoT
scripts remain available but have not been validated as part of this release.

An unresolved upstream merge marker in the unused IMDb tool was removed so all
bundled Python sources can compile.

Two nested-quote f-strings in the StruQ backend and weather tool were made
compatible with the declared Python 3.11 environment.

The optional `data/agent_task_pot_all.jsonl` from the source copy was an HTML
rate-limit page, so it is excluded from this release. Standard DPI data is validated JSONL.
