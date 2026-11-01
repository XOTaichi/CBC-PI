"""Evaluate frozen GCG suffixes using the author's original prompt construction."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    expanded = os.path.expanduser(os.path.expandvars(value))
    if '$' in expanded:
        raise ValueError(f'Unresolved environment variable: {value}')
    return Path(expanded).resolve()


def load_inputs(path):
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError('Input must be a non-empty JSON array')
    for index, item in enumerate(data):
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str)
                                             for k in ('instruction', 'input', 'goal')):
            raise ValueError(f'Invalid instruction/input/goal at row {index}')
        if not isinstance(item.get('best_string', ''), str):
            raise ValueError(f'Invalid best_string at row {index}')
    return data


def build_model(args, model_path):
    from models import VLLMModel, VLLMDataModel, Sandwich, PromptArmor
    common = dict(tensor_parallel_size=args.tensor_parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=args.max_model_len)
    if args.model_type == 'vllm_data':
        model = VLLMDataModel(str(model_path), role='data', **common)
    else:
        model = VLLMModel(str(model_path), **common)
    model.sampling_params.temperature = args.temperature
    if args.model_type == 'sandwich':
        return Sandwich(model)
    if args.model_type == 'promptarmor':
        return PromptArmor(model)
    return model


def process_file(data, output_path, model, n_responses):
    syss = [item['instruction'] for item in data]
    # Preserve original concatenation, including the absence of a separator before best_string.
    prompts = [item['input'] + '\n' + item['goal'] + item.get('best_string', '') for item in data]
    results = [{k: item.get(k, '') for k in ('instruction', 'input', 'goal', 'best_string')} for item in data]
    for item in results:
        item['outputs'] = []
    for run in range(n_responses):
        print(f'Generating response {run + 1}/{n_responses}', flush=True)
        # Keep the original system/user interface even for vllm_data; do not silently change the protocol.
        outputs = model.batch_generate_with_system(syss, prompts)
        if len(outputs) != len(results) or any(not isinstance(x, str) for x in outputs):
            raise ValueError('Model returned invalid outputs or an unexpected number of responses')
        for item, response in zip(results, outputs):
            item['outputs'].append(response)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental overwrites, including concurrent runs.
    with output_path.open('x') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved {len(results)} examples to {output_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Frozen attack JSON file')
    parser.add_argument('--output', required=True, help='New generation-results JSON file')
    parser.add_argument('--model-path', required=True, help='Local checkpoint directory')
    parser.add_argument('--model-type', choices=['vllm', 'vllm_data', 'sandwich', 'promptarmor'], default='vllm')
    parser.add_argument('--n-responses', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=9000)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    parser.add_argument('--promptarmor-model', help='Filter model; overrides PROMPTARMOR_MODEL')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    load_dotenv(ROOT / '.env')
    if args.n_responses < 1 or args.tensor_parallel_size < 1 or args.max_model_len < 1:
        parser.error('Response count, tensor parallel size, and context length must be positive')
    if not 0 < args.gpu_memory_utilization <= 1 or args.temperature < 0:
        parser.error('GPU utilization must be in (0, 1] and temperature must be non-negative')
    input_path, output_path, model_path = map(resolve_path, (args.input, args.output, args.model_path))
    data = load_inputs(input_path)
    if not model_path.is_dir():
        parser.error(f'Model directory does not exist: {model_path}')
    if output_path.exists():
        parser.error(f'Output already exists: {output_path}')
    if args.promptarmor_model is not None:
        if not args.promptarmor_model.strip():
            parser.error('--promptarmor-model must not be blank')
        os.environ['PROMPTARMOR_MODEL'] = args.promptarmor_model.strip()
    print(f'Input: {input_path}\nModel: {args.model_type}::{model_path}\n'
          f'Examples: {len(data)}; responses per example: {args.n_responses}\nOutput: {output_path}')
    if args.dry_run:
        return
    if args.model_type == 'promptarmor' and not all(os.getenv(k) for k in ('ARK_API_KEY', 'ARK_BASE_URL')):
        parser.error('PromptArmor requires ARK_API_KEY and ARK_BASE_URL')
    process_file(data, output_path, build_model(args, model_path), args.n_responses)


if __name__ == '__main__':
    main()
