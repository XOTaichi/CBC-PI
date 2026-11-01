# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import os
from typing import List, Optional
import re
import vllm
from vllm import SamplingParams
from typing_extensions import override
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ..benchmark_utils import audio_to_b64, image_to_b64
from .llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
)


class VLLM(LLM):
    """Accessing VLLM"""

    def __init__(self, config: LLMConfig) -> None:

        super().__init__(config)
        model_path = config.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.base_model = vllm.LLM(model=model_path, dtype=torch.bfloat16, tensor_parallel_size=int(os.getenv("CBC_TENSOR_PARALLEL_SIZE", "1")),
                    gpu_memory_utilization=float(os.getenv("CBC_GPU_MEMORY_UTILIZATION", "0.7")),
                    enforce_eager=os.getenv("CBC_ENFORCE_EAGER", "0") == "1",
                    **({"max_model_len": int(os.environ["CBC_MAX_MODEL_LEN"])} if os.getenv("CBC_MAX_MODEL_LEN") else {}))
        self.sampling_params = SamplingParams(
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=int(os.getenv("CBC_MAX_NEW_TOKENS", "4096"))
        )
    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.base_model.generate(text, self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.base_model.generate(text, self.sampling_params)
        logging.getLogger().setLevel(level)
        return outputs[0].outputs[0].text.strip()

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.base_model.generate(text, self.sampling_params)
        logging.getLogger().setLevel(level)
        return outputs[0].outputs[0].text.strip()

    def extract_answer(self, text):
        match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            think_pos = text.lower().find('</think>')
            if think_pos != -1:
                return text[think_pos + len('</think>'):].strip()
            else:
                return text.strip()

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = [ {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(text)
        outputs = self.base_model.generate(text, self.sampling_params)
        logging.getLogger().setLevel(level)
        return self.extract_answer(outputs[0].outputs[0].text.strip())
