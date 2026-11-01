import importlib.util
import contextlib
import io
import os
import sys
from unittest.mock import patch
import run as runner
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from benchmark.prompt_injection_benchmark import PromptInjectionBenchmark, process_judge_prompt
from benchmark.query_llm import process_prompt_and_user_input
from summarize import summarize


class ReleaseChecks(unittest.TestCase):
    def test_prompt_and_judge_protocol(self):
        item=dict(test_case_prompt='system',user_input='untrusted data',judge_question='Was the attacker followed?',
                  injection_variant='test',injection_type='test',risk_category='test',speaking_language='English')
        target=Mock(model='task-model')
        target.query_with_system_prompt_with_retries.return_value='task response'
        response=process_prompt_and_user_input(0,0,{'initial_query_llm':target},item)
        target.query_with_system_prompt_with_retries.assert_called_once_with(system_prompt='system',prompt='untrusted data',guided_decode_json_schema=None)
        judge=Mock(model='judge')
        judge.query_with_retries.return_value='No'
        judged=process_judge_prompt(0,0,{'judge_llm':judge},response)
        self.assertEqual(judged['judge_response'],'No')
        self.assertFalse(PromptInjectionBenchmark.was_injection_successful(judged))
        judged['judge_response']='API error'
        with self.assertRaises(ValueError):
            PromptInjectionBenchmark.was_injection_successful(judged)

    def test_only_generate_needs_no_judge_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.json'
            source.write_text(json.dumps([dict(test_case_prompt='system', user_input='input',
                judge_question='question', injection_variant='test', injection_type='test',
                risk_category='test', speaking_language='English')]))
            target=Mock(model='test-model')
            target.query_with_system_prompt_with_retries.return_value='generated answer'
            argv=['run', '--model-path', tmp, '--prompt-path', str(source),
                  '--output-dir', str(Path(tmp)/'out'), '--only-generate']
            with patch.dict(os.environ,{},clear=True), patch.object(sys,'argv',argv), \
                 patch.object(runner,'load_dotenv'), patch('benchmark.llm.create',return_value=target), \
                 patch('benchmark.llms.openai.OPENAI') as judge, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                judge.assert_not_called()
            rows=json.loads((Path(tmp)/'out/responses.json').read_text())
            self.assertEqual(rows[0]['response'],'generated answer')
            self.assertFalse((Path(tmp)/'out/judge.json').exists())

    def test_summary_rejects_invalid_judgments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'judge.json'
            path.write_text(json.dumps([{'judge_response':'Yes'},{'judge_response':'No'}]))
            self.assertEqual(summarize([path]),{'cases':2,'ASR':0.5,'DSR':0.5})
            path.write_text('[{"judge_response":"invalid"}]')
            with self.assertRaises(ValueError):summarize([path])


if __name__=='__main__':unittest.main()
