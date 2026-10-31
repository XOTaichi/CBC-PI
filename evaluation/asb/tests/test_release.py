"""CPU-only checks for release plumbing; no model loading or API calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load('agent_attack')
summary = load('summarize')


class ReleaseChecks(unittest.TestCase):
    def test_dry_run_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model with spaces'
            model.mkdir()
            argv = ['agent_attack', '--cfg_path', str(ROOT / 'config/qwen2/DPI_RL.yml'),
                    '--model', f'vllm::{model}', '--dry-run']
            with patch.object(sys, 'argv', argv), patch.object(launcher.subprocess, 'run') as run:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    launcher.main()
                self.assertIn('Planned cases:', out.getvalue())
                run.assert_not_called()

    def test_unset_variable_is_rejected(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                launcher.expand('${CBC_MODELS_DIR}/checkpoint')

    def test_child_failure_is_propagated(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['agent_attack', '--cfg_path', str(ROOT / 'config/qwen2/DPI_RL.yml'),
                    '--model', f'vllm::{tmp}', '--output_dir', f'{tmp}/out', '--skip-refusal-judge']
            with patch.object(sys, 'argv', argv), patch.object(launcher.subprocess, 'run',
                    side_effect=subprocess.CalledProcessError(1, 'evaluator')) as run:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                    launcher.main()
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertEqual(run.call_args.kwargs['cwd'], ROOT)

    def test_summary_missing_and_invalid_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample.csv'
            path.write_text('Attack Successful,Original Task Successful,Refuse Result\n1,0,N/A\n0,1,N/A\n')
            result = summary.summarize(path)
            self.assertEqual(result['ASR'], 0.5)
            self.assertEqual(result['DSR'], 0.5)
            self.assertEqual(result['original_task_success_rate'], 0.5)
            self.assertIsNone(result['refusal_rate'])
            path.write_text('Attack Successful,Original Task Successful,Refuse Result\n0,1,ERROR\n')
            with self.assertRaises(ValueError):
                summary.summarize(path)


if __name__ == '__main__':
    unittest.main()
