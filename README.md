# CBC: Consistency Before Complying

Official repository for **Staying on Task: Defending Against Prompt Injection via Task Consistency Checking**, accepted to **EMNLP 2026**.

**Feiyue Xu, Jiaming Mu, Fengting Li, Bin Benjamin Zhu, Hongsheng Hu, Qiankun Liu, Shuo Wang, Chaoxiang He, and Chao Feng**

[Paper](https://aclanthology.org/2026.emnlp-main.2303/) · [Citation](#citation) · [Overview](#overview) · [Models](#model-downloads) · [Datasets](#dataset-downloads) · [Evaluation](#evaluation) · [Data synthesis](#cbc-data-synthesis-todo) · [TODO](#release-roadmap)

## Overview

**CBC (Consistency Before Complying)** is a task-consistency alignment framework for defending LLMs against task-redirection prompt injection. Before following a newly introduced instruction, CBC reasons about whether it is consistent with the intended task.

CBC checks semantic conflicts across **actions, constraints, and domains**, then follows a structured process: identify instructions and the main task, assess task consistency, and select a response policy. Training combines synthetic supervision for structured consistency reasoning with dual-objective online alignment that rewards both injection resistance and task utility. At inference time, the model performs consistency reasoning directly, without a separate external judge.

<p align="center">
  <img src="assets/cbc-overview.png" alt="CBC task-consistency reasoning and training pipeline, including semantic projection, cold-start supervision, and online GRPO alignment." width="900">
</p>

*Figure 2 from the paper. CBC combines task-consistency reasoning (top), cold-start supervision with structured reasoning (bottom left), and online GRPO alignment for task utility and injection resistance (bottom right).*

Experiments cover three backbones: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Qwen3-14B. Across direct and indirect prompt-injection benchmarks, CBC improves defense success rates by up to **13.2 percentage points** over the strongest baseline while better preserving task utility under attack.

## Release roadmap

This repository is being prepared for release. Evaluation entry points are listed below for reference; their migration into this repository and standalone reproduction instructions are pending.

- [x] Publish the project overview and method illustration.
- [ ] Release evaluation code, environment specifications, and reproduction commands.
- [x] Publish CBC model checkpoints and download links.
- [ ] Publish CBC training data and the exact evaluation subsets/configurations.
- [ ] **Release the pipeline for synthesizing CBC training data from scratch.**

## Model downloads

### CBC checkpoints

| Backbone | CBC checkpoint |
| --- | --- |
| Qwen2.5-7B-Instruct | [augusttailong/grpo_qwen_0202](https://modelscope.cn/models/augusttailong/grpo_qwen_0202) |
| Llama-3.1-8B-Instruct | [augusttailong/grpo_llama3_0204](https://modelscope.cn/models/augusttailong/grpo_llama3_0204) |
| Qwen3-14B | [augusttailong/grpo_qwen3_0211](https://modelscope.cn/models/augusttailong/grpo_qwen3_0211) |

CBC checkpoints are hosted on ModelScope. Inference examples will accompany the evaluation code release. The following links provide the original backbones; they do not contain CBC alignment weights.

### Original backbones

| Model | Download |
| --- | --- |
| Qwen2.5-7B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Llama-3.1-8B-Instruct | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen3-14B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-14B) |

Please follow the access requirements and license terms on each model page.

## Dataset downloads

### CBC training data

| Dataset | Contents | Download |
| --- | --- | --- |
| CBC-SFT | Synthetic task-consistency supervision with structured reasoning; 23,637 training and 1,243 validation examples after filtering | Coming soon |
| CBC-GRPO | Online-alignment prompts; 4,880 training and 1,220 validation examples | Coming soon |

These sizes follow Appendix C.2 of the paper. The data-synthesis code is a separate release item; see [CBC data synthesis](#cbc-data-synthesis-todo).

### Prompt-injection evaluation data

| Benchmark | Setting | Source / download instructions |
| --- | --- | --- |
| SEP | Indirect injection | [Official code and dataset](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed) |
| OpenPrompt | Indirect injection | [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) |
| InjecAgent | Indirect injection, enhanced setting | [Official code and data](https://github.com/uiuc-kang-lab/InjecAgent) |
| LlamaCyberEval | Direct injection, prompt-injection subset | [PurpleLlama / CybersecurityBenchmarks](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks) |
| ASB | Direct injection and task utility under attack | [Agent Security Bench](https://github.com/agiresearch/ASB) |

These are upstream resources. Exact paper subsets and sampling configurations will be documented with the evaluation release; downloading an upstream dataset alone does not reproduce the paper's evaluation split.

### General task utility

Utility evaluation covers [MMLU](https://huggingface.co/datasets/cais/mmlu), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [BBQ](https://github.com/nyu-mll/BBQ), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), and [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval).

Our general task utility evaluation builds on [DeepEval](https://github.com/confident-ai/deepeval), with minor modifications to support evaluation under prompt-injection settings. We thank the DeepEval contributors for their open-source evaluation framework.

## Evaluation

Awaiting update (October 31, 2026).

## CBC data synthesis (TODO)

Awaiting update (October 31, 2026).

## Citation

If you find this work useful, please cite our [EMNLP 2026 paper](https://aclanthology.org/2026.emnlp-main.2303/). You can also download the complete [BibTeX entry](citation.bib).

```bibtex
@inproceedings{xu-etal-2026-staying,
    title = "Staying on Task: Defending Against Prompt Injection via Task Consistency Checking",
    author = "Xu, Feiyue  and
      Mu, Jiaming  and
      Li, Fengting  and
      Zhu, Bin Benjamin  and
      Hu, Hongsheng  and
      Liu, Qiankun  and
      Wang, Shuo  and
      He, Chaoxiang  and
      Feng, Chao",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2303/",
    doi = "10.18653/2026.emnlp-main.2303",
    pages = "52349--52375",
    ISBN = "979-8-89176-459-0"
}
```

## Acknowledgments

We thank the authors of the models, benchmarks, and datasets linked above. Third-party resources remain subject to their respective licenses.
