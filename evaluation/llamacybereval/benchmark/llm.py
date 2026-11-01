# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in ../LICENSE.
"""Model factory for the prompt-injection release subset."""
import os
from .llms.llm_base import (
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, LLM, LLMConfig,
    MAX_RETRY_TIMEOUT, NUM_LLM_RETRIES, RetriesExceededException, UnretriableQueryException,
)

SPECIFICATION_FORMAT = '<PROVIDER>::<MODEL>(::<API_KEY>)?(::<BASE_URL>)?'
EXAMPLE_SPECIFICATION = 'VLLM::/absolute/path/to/checkpoint'


def create(identifier, max_tokens=DEFAULT_MAX_TOKENS, cot_end_tokens=None):
    parts = identifier.split('::')
    provider = parts[0]
    if provider.startswith('PE_'):
        if len(parts) != 5:
            raise ValueError('Prompt wrappers use PE_NAME::VLLM::MODEL::x::x')
        _, wrapped_provider, name, key, url = parts
        base = create(f'{wrapped_provider}::{name}' + (f'::{key}::{url}' if wrapped_provider == 'OPENAI' else ''),
                      max_tokens, cot_end_tokens)
        from .llms.prompt_engineering import Sandwich, Instruction, Instruction_cot, PromptArmor
        wrappers = {'PE_SANDWICH': Sandwich, 'PE_INSTRUCTION': Instruction,
                    'PE_INSTRUCTION_COT': Instruction_cot, 'PE_PROMPTARMOR': PromptArmor}
        if provider not in wrappers:
            raise ValueError(f'Unsupported prompt wrapper: {provider}')
        return wrappers[provider](config=LLMConfig(model='pe'), LLM2=base)
    if len(parts) not in (2, 3, 4):
        raise ValueError('Expected PROVIDER::MODEL[::API_KEY[::BASE_URL]]')
    name = parts[1]
    config = LLMConfig(model=name, api_key=parts[2] if len(parts)>2 else os.getenv('OPENAI_API_KEY'),
                       base_url=parts[3] if len(parts)>3 else os.getenv('OPENAI_BASE_URL'),
                       max_tokens=max_tokens, cot_end_tokens=cot_end_tokens)
    if provider == 'VLLM':
        from .llms.vllm_model import VLLM
        return VLLM(config)
    if provider == 'OPENAI':
        from .llms.openai import OPENAI
        return OPENAI(config)
    raise ValueError(f'Unsupported provider in this release: {provider}; use VLLM or OPENAI')
