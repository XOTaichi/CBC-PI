"""Launch the original ASB evaluator with portable, validated configuration."""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {'all': 'all_attack_tools.jsonl', 'agg': 'all_attack_tools_aggressive.jsonl',
         'non-agg': 'all_attack_tools_non_aggressive.jsonl', 'test': 'attack_tools_test.jsonl'}
INJECTIONS = {
    'direct_prompt_injection': ['--direct_prompt_injection'],
    'observation_prompt_injection': ['--observation_prompt_injection'],
    'memory_attack': ['--memory_attack'], 'clean': ['--clean'],
    'mixed_attack': ['--direct_prompt_injection', '--observation_prompt_injection'],
    'DPI_OPI': ['--direct_prompt_injection', '--observation_prompt_injection'],
    'DPI_MP': ['--direct_prompt_injection'], 'OPI_MP': ['--observation_prompt_injection'],
}


def expand(value):
    value = os.path.expandvars(value)
    if re.search(r'\$\{?\w+', value):
        raise ValueError(f'Unresolved environment variable in: {value}')
    return os.path.expanduser(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_path', required=True)
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--model', help='Override config models, e.g. vllm::/path/to/checkpoint')
    parser.add_argument('--promptarmor-model', help='Override the PromptArmor filter model')
    parser.add_argument('--task_num', type=int, help='Tasks per agent (original default: 1)')
    parser.add_argument('--tasks_path', help='Override task JSONL')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without loading a model or calling APIs')
    parser.add_argument('--skip-refusal-judge', action='store_true', help='Write N/A for refusal metric; ASR and utility still computed')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    cfg_path = Path(args.cfg_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    method = cfg['injection_method']
    if method not in INJECTIONS:
        raise ValueError(f'Unsupported injection method: {method}')
    if args.task_num is not None and args.task_num < 1:
        raise ValueError('--task_num must be positive')
    task_path = Path(expand(args.tasks_path or cfg.get('tasks_path', str(ROOT / 'data/agent_task.jsonl')))).resolve()
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if not tasks:
        raise ValueError('Task file is empty')
    models = [args.model] if args.model else cfg['llms']
    for tool_type, model, attack in itertools.product(cfg['attack_tool'], models, cfg['attack_types']):
        model = expand(model)
        backend = None
        if '::' in model:
            backend, model = model.split('::', 1)
            if backend not in {'vllm', 'cot', 'sandwich', 'promptarmor'}:
                raise ValueError(f'Unknown backend: {backend}')
            if model.startswith(('/', '.')) and not Path(model).is_dir():
                raise ValueError(f'Model directory does not exist: {model}')
        elif model.startswith('ollama/'):
            backend, model = 'ollama', model.removeprefix('ollama/')
        elif not model.startswith(('gpt', 'gemini', 'claude')):
            raise ValueError('Use an explicit backend, e.g. vllm::/path/to/model')
        tool_path = ROOT / 'data' / TOOLS[tool_type]
        tools = [json.loads(line) for line in tool_path.read_text().splitlines() if line.strip()]
        count = sum(min(len(task['tasks']), args.task_num or 1) *
                    sum(tool['Corresponding Agent'] == task['agent_name'] for tool in tools)
                    for task in tasks)
        if not count:
            raise ValueError('No task/attacker pairs match the selected files')
        label = re.sub(r'[^\w.-]+', '_', f'{backend}_{model}')
        mode = cfg.get('defense_type') or ('new_memory' if cfg.get('read_db') else 'no_memory')
        stem = f"{attack}-{tool_type}_{cfg.get('suffix', '')}"
        if any('/' in str(v) or '\\' in str(v) or str(v) in {'.', '..'} for v in (mode, stem)):
            raise ValueError('Defense, attack and suffix must be filename components')
        output = Path(args.output_dir).resolve() / method / label / mode
        csv_path = output / f'{stem}.csv'
        log_path = output / f'{stem}.log'
        cmd = [sys.executable, str(ROOT / 'main_attacker.py'), '--llm_name', model,
               '--attack_type', attack, '--attacker_tools_path', str(tool_path),
               '--tasks_path', str(task_path), '--res_file', str(csv_path)]
        if backend:
            cmd += ['--use_backend', backend]
        for flag in ('read_db', 'write_db'):
            if cfg.get(flag):
                cmd.append(f'--{flag}')
        if cfg.get('read_db') or cfg.get('write_db'):
            database = ROOT / 'memory_db/direct_prompt_injection' / f'{attack}_gpt-4o-mini'
            if not database.is_dir():
                raise ValueError('Memory database is not bundled; prepare it before memory evaluation')
            cmd += ['--database', str(database)]
        if cfg.get('defense_type'):
            cmd += ['--defense_type', cfg['defense_type']]
        if args.task_num:
            cmd += ['--task_num', str(args.task_num)]
        if args.skip_refusal_judge:
            cmd.append('--skip_refusal_judge')
        cmd += INJECTIONS[method]
        child_env = os.environ.copy()
        if backend == 'promptarmor':
            filter_model = (args.promptarmor_model if args.promptarmor_model is not None else
                            cfg.get('promptarmor_model', os.getenv('PROMPTARMOR_MODEL', 'kimi-k2.6')))
            if not isinstance(filter_model, str) or not filter_model.strip():
                raise ValueError('promptarmor_model must be a non-empty model identifier')
            child_env['PROMPTARMOR_MODEL'] = filter_model.strip()
            print(f'PromptArmor filter model: {filter_model.strip()}', flush=True)
        print(f'Planned cases: {count}\n{shlex.join(cmd)}\nLog: {log_path}', flush=True)
        if args.dry_run:
            continue
        if not args.skip_refusal_judge and not all(os.getenv(k) for k in ('GLM_API_KEY', 'GLM_BASE_URL')):
            raise ValueError('Set GLM_API_KEY and GLM_BASE_URL for refusal judging, or explicitly use --skip-refusal-judge')
        if csv_path.exists() or log_path.exists():
            raise FileExistsError(f'Output already exists: {output}; choose another --output_dir or suffix')
        output.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=child_env)


if __name__ == '__main__':
    main()
