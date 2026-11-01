import os
import sys
import json
import argparse

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import get_score


def find_result_files(result_dir):
    files = [f for f in os.listdir(result_dir) if f.endswith('.jsonl')]
    dh = [f for f in files if '_dh_' in f]
    ds = [f for f in files if '_ds_' in f]
    return dh, ds


def eval_single_dir(result_dir):
    dh, ds = find_result_files(result_dir)
    if not dh or not ds:
        print(f"Error: missing dh or ds jsonl in {result_dir}")
        sys.exit(1)

    output_files = {
        'dh': os.path.join(result_dir, dh[0]),
        'ds': os.path.join(result_dir, ds[0])
    }
    scores = get_score(output_files)
    print(json.dumps(scores, indent=2))


def convert_asr_to_dsr(scores):
    """Convert ASR-all (Total) to DSR = 100 - ASR."""
    scores = dict(scores)
    key = "ASR-all (Total)"
    if key in scores:
        val = scores[key]
        if val != '-':
            try:
                asr = float(val)
                dsr = 100.0 - asr
                scores["DSR (Total)"] = f"{dsr:.1f}"
            except ValueError:
                scores["DSR (Total)"] = '-'
        else:
            scores["DSR (Total)"] = '-'
    return scores


def eval_all_under(base_dir):
    """Evaluate all model/run directories and print per-run DSR in a table."""
    model_results = {}

    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        run_dirs = []
        for entry in sorted(os.listdir(model_path)):
            entry_path = os.path.join(model_path, entry)
            if os.path.isdir(entry_path) and entry.lower().startswith('run'):
                run_dirs.append((entry, entry_path))

        if not run_dirs:
            # Try evaluating the model dir itself if no run subdirs
            dh, ds = find_result_files(model_path)
            if dh and ds:
                run_dirs.append(('.', model_path))

        if not run_dirs:
            continue

        runs = {}
        for run_name, run_path in run_dirs:
            # Search recursively under run_path for jsonl files
            dh, ds = find_result_files(run_path)
            result_path = run_path
            if not dh or not ds:
                for root, dirs, files in os.walk(run_path):
                    dh, ds = find_result_files(root)
                    if dh and ds:
                        result_path = root
                        break
                else:
                    continue
            if not dh or not ds:
                continue
            output_files = {
                'dh': os.path.join(result_path, dh[0]),
                'ds': os.path.join(result_path, ds[0])
            }
            scores = get_score(output_files)
            scores = convert_asr_to_dsr(scores)
            # Extract numeric DSR for summary table
            dsr_str = scores.get("DSR (Total)", "-")
            try:
                dsr_val = float(dsr_str)
            except (ValueError, TypeError):
                dsr_val = None
            runs[run_name] = {
                'scores': scores,
                'dsr': dsr_val
            }

        if runs:
            model_results[model_name] = runs

    if not model_results:
        print("No results found.")
        return

    # Print per-run detailed scores
    for model_name, runs in sorted(model_results.items()):
        print(f"\n=== {model_name} ===")
        for run_name in sorted(runs.keys(), key=lambda x: (len(x), x)):
            print(f"  --- {run_name} ---")
            print(json.dumps(runs[run_name]['scores'], indent=4))

    # Print summary table: model x runs DSR
    all_runs = set()
    for runs in model_results.values():
        all_runs.update(runs.keys())
    run_headers = sorted(all_runs, key=lambda x: (len(x), x))

    print("\n" + "=" * 80)
    print("DSR Summary Table (DSR = 100 - ASR-all (Total))")
    print("=" * 80)

    header = f"{'Model':<30}"
    for run_name in run_headers:
        header += f" {run_name:>10}"
    print(header)
    print("-" * len(header))

    for model_name in sorted(model_results.keys()):
        line = f"{model_name:<30}"
        for run_name in run_headers:
            dsr = model_results[model_name].get(run_name, {}).get('dsr')
            if dsr is not None:
                line += f" {dsr:>10.1f}"
            else:
                line += f" {'-':>10}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Evaluate InjecAgent results without loading a model.")
    parser.add_argument('path', help='Result directory (or base directory with --all)')
    parser.add_argument('--all', action='store_true', help='Recursively evaluate all result directories under path')
    args = parser.parse_args()

    if args.all:
        eval_all_under(args.path)
    else:
        eval_single_dir(args.path)


if __name__ == '__main__':
    main()
