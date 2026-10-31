"""Check filter-model selection without loading weights or contacting an API."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_release import launcher
from aios.llm_core.llm_classes.promptarmor_vllm import PromptArmor_vLLM


class PromptArmorChecks(unittest.TestCase):
    def test_filter_model_and_request(self):
        for selected in (None, 'my-filter-model'):
            with self.subTest(selected=selected), patch.dict(os.environ, {}, clear=True):
                if selected is not None:
                    os.environ['PROMPTARMOR_MODEL'] = selected
                wrapper = PromptArmor_vLLM.__new__(PromptArmor_vLLM)
                wrapper.model_name = 'unused-task-checkpoint'
                wrapper.max_new_tokens = 4096
                with patch('vllm.LLM'), patch('transformers.AutoTokenizer.from_pretrained'), patch('openai.OpenAI') as client:
                    wrapper.load_llm_and_tokenizer()
                    client.return_value.chat.completions.create.return_value = SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content='Yes\nInjection: malicious'))])
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(wrapper._filter('original task', 'safe malicious'), (True, 'safe '))
                    self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs['model'],
                                     selected or 'kimi-k2.6')

    def test_cli_yaml_environment_precedence(self):
        for yaml_model, cli_model, expected in ((None, None, 'environment-model'),
                ('yaml-model', None, 'yaml-model'), ('yaml-model', 'cli-model', 'cli-model')):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                cfg = Path(tmp) / 'config.yml'
                cfg.write_text(f'injection_method: direct_prompt_injection\nattack_tool: [all]\n'
                               f'llms: ["promptarmor::{tmp}"]\nattack_types: [combined_attack]\n' +
                               (f'promptarmor_model: {yaml_model}\n' if yaml_model else ''))
                argv = ['agent_attack', '--cfg_path', str(cfg), '--output_dir', f'{tmp}/out', '--skip-refusal-judge']
                if cli_model:
                    argv += ['--promptarmor-model', cli_model]
                with patch.dict(os.environ, {'PROMPTARMOR_MODEL': 'environment-model'}), \
                        patch.object(sys, 'argv', argv), patch.object(launcher.subprocess, 'run') as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        launcher.main()
                    self.assertEqual(run.call_args.kwargs['env']['PROMPTARMOR_MODEL'], expected)
                    self.assertEqual(os.environ['PROMPTARMOR_MODEL'], 'environment-model')

    def test_blank_yaml_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'config.yml'
            cfg.write_text(f'injection_method: direct_prompt_injection\nattack_tool: [all]\n'
                           f'llms: ["promptarmor::{tmp}"]\nattack_types: [combined_attack]\n'
                           'promptarmor_model: " "\n')
            argv = ['agent_attack', '--cfg_path', str(cfg), '--dry-run']
            with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                launcher.main()


if __name__ == '__main__':
    unittest.main()
