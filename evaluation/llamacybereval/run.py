"""Portable entry point for CBC's LlamaCyberEval prompt-injection subset."""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def validate_prompts(path):
    rows = json.loads(path.read_text())
    required = ('test_case_prompt', 'user_input', 'judge_question', 'injection_variant',
                'injection_type', 'risk_category', 'speaking_language')
    if not isinstance(rows, list) or not rows or any(any(k not in row for k in required) for row in rows):
        raise ValueError(f'Invalid prompt-injection JSON: {path}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-type', choices=['VLLM', 'SANDWICH', 'PROMPTARMOR'], default='VLLM')
    p.add_argument('--prompt-path', default=str(ROOT/'datasets/prompt_injection/prompt_injection.json'))
    p.add_argument('--output-dir', required=True)
    p.add_argument('--judge-model', default=None)
    p.add_argument('--num-test-cases', type=int, default=0, help='0 uses the full selected file')
    p.add_argument('--only-generate', action='store_true', help='Generate responses without an external judge')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    load_dotenv(ROOT/'.env')
    model = Path(os.path.expandvars(os.path.expanduser(args.model_path))).resolve()
    source = Path(args.prompt_path).resolve()
    output = Path(args.output_dir).resolve()
    if not model.is_dir():
        p.error(f'Model directory does not exist: {model}')
    rows = validate_prompts(source)
    if args.num_test_cases < 0:
        p.error('--num-test-cases must be non-negative')
    if any((output/name).exists() for name in ['responses.json','judge.json','stats.json']):
        p.error('Output already exists; choose a new --output-dir')
    print(f'{source.name}: {min(len(rows), args.num_test_cases or len(rows))} cases; model={args.model_type}::{model}')
    if args.dry_run:
        return
    key = os.getenv('JUDGE_API_KEY') or os.getenv('ARK_API_KEY') or os.getenv('OPENAI_API_KEY')
    url = os.getenv('JUDGE_BASE_URL') or os.getenv('ARK_BASE_URL') or os.getenv('OPENAI_BASE_URL')
    if not key and not args.only_generate:
        p.error('Set JUDGE_API_KEY (or ARK_API_KEY / OPENAI_API_KEY) for response judging')
    from benchmark import llm
    from benchmark.benchmark import BenchmarkConfig
    from benchmark.prompt_injection_benchmark import PromptInjectionBenchmark
    from benchmark.llms.openai import OPENAI
    from benchmark.llms.llm_base import LLMConfig
    import asyncio
    identifier = f'VLLM::{model}' if args.model_type == 'VLLM' else f'PE_{args.model_type}::VLLM::{model}::x::x'
    if args.model_type == 'PROMPTARMOR' and not all(os.getenv(k) for k in ['ARK_API_KEY','ARK_BASE_URL']):
        p.error('PromptArmor requires ARK_API_KEY and ARK_BASE_URL')
    judge = None if args.only_generate else OPENAI(LLMConfig(model=args.judge_model or os.getenv('JUDGE_MODEL','deepseek-v3.2'),
                             api_key=key, base_url=url, max_tokens=2048))
    target = llm.create(identifier)
    output.mkdir(parents=True, exist_ok=True)
    if args.only_generate:
        from benchmark.query_llm import query_llm_to_generate_responses, process_prompt_and_user_input
        responses = query_llm_to_generate_responses({'initial_query_llm': target}, source,
                    to_process_prompt=process_prompt_and_user_input, num_test_cases=args.num_test_cases)
        if any(not isinstance(row.get('response'), str) or not row['response'].strip() for row in responses):
            raise ValueError('Model returned an empty response')
        (output/'responses.json').write_text(json.dumps(responses, indent=2))
        print(f'Generated {len(responses)} responses; external judging skipped.')
        return
    benchmark = PromptInjectionBenchmark(BenchmarkConfig(llms_under_test=[target], judge_llm=judge,
                    response_path=output/'responses.json', judge_response_path=output/'judge.json',
                    stat_path=output/'stats.json', num_test_cases=args.num_test_cases))
    benchmark.query_llm_to_generate_responses(source, run_llm_in_parallel=1)
    asyncio.run(benchmark.run(run_llm_in_parallel=1))


if __name__ == '__main__':
    main()
