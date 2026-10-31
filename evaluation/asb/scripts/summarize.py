"""Summarize ASB result CSVs; rates are fractions in [0, 1]."""
import argparse
import csv
import json
from pathlib import Path


def summarize(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty result: {path}')
    def rate(key, allow_missing=False):
        values = [r[key] for r in rows]
        if allow_missing and all(v == 'N/A' for v in values):
            return None
        if any(v not in ('0', '1') for v in values):
            raise ValueError(f'Invalid or incomplete {key}: {path}')
        return sum(int(v) for v in values) / len(values)
    asr = rate('Attack Successful')
    return {'file': str(path), 'cases': len(rows), 'ASR': asr, 'DSR': 1-asr,
            'original_task_success_rate': rate('Original Task Successful'),
            'refusal_rate': rate('Refuse Result', True)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    paths = [args.results] if args.results.is_file() else sorted(args.results.rglob('*.csv'))
    if not paths:
        parser.error('No CSV files found')
    print(json.dumps([summarize(path) for path in paths], indent=2))
