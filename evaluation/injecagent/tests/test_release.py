import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils import get_score, get_simulated_attacker_tool_response
from src.evaluate_prompted_agent import main
from src.eval_only import convert_asr_to_dsr

ROOT=Path(__file__).resolve().parents[1]


class ReleaseChecks(unittest.TestCase):
    def test_scores_need_no_api(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            dh=Path(tmp)/'dh.jsonl';ds=Path(tmp)/'ds.jsonl'
            dh.write_text('{"eval":"succ"}\n{"eval":"unsucc"}\n')
            ds.write_text('{"eval":"succ","eval Step 2":"unsucc"}\n{"eval":"unsucc"}\n')
            scores=get_score({'dh':str(dh),'ds':str(ds)})
            self.assertEqual(scores['#Test Case'],4)
            self.assertEqual(scores['ASR-all (Total)'],'25.0')
            self.assertEqual(convert_asr_to_dsr(scores)['DSR (Total)'],'75.0')

    def test_cached_tool_response_needs_no_api(self):
        cache=json.loads((ROOT/'data/attacker_simulated_responses.json').read_text())
        key,value=next(iter(cache.items()))
        tool,params=key[1:-1].split(',',1)
        with patch('src.utils.get_simulator_client') as client:
            self.assertEqual(get_simulated_attacker_tool_response(tool,params,{}),value)
            client.assert_not_called()

    def test_case_limit_applies_to_each_attack_type(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as out:
            main(dict(model_type='VLLM',model_name='unused-checkpoint',output_dir=tmp,
                      setting='enhanced',prompt_type='InjecAgent',only_get_score=False,dry_run=True,max_cases=1))
            self.assertEqual(out.getvalue().count('1 cases'),2)

    def test_dry_run_validates_without_output(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()) as out:
            output=Path(tmp)/'output'
            main(dict(model_type='VLLM',model_name='unused-checkpoint',output_dir=str(output),
                      setting='enhanced',prompt_type='InjecAgent',only_get_score=False,dry_run=True))
            self.assertIn('510 cases',out.getvalue())
            self.assertIn('544 cases',out.getvalue())
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
