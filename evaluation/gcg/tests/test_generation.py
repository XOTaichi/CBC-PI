import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import batch_generate as generate


class GenerationChecks(unittest.TestCase):
    def test_original_prompt_and_response_order(self):
        data = [{'instruction': 'system', 'input': 'data', 'goal': 'goal', 'best_string': 'suffix'}]
        model = Mock()
        model.batch_generate_with_system.side_effect = [['first'], ['second'], ['third']]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp) / 'results.json'
            generate.process_file(data, output, model, 3)
            self.assertEqual(json.loads(output.read_text())[0]['outputs'], ['first', 'second', 'third'])
            model.batch_generate_with_system.assert_called_with(['system'], ['data\ngoalsuffix'])
            with self.assertRaises(FileExistsError):
                generate.process_file(data, output, Mock(batch_generate_with_system=Mock(return_value=['new'])), 1)
            self.assertEqual(json.loads(output.read_text())[0]['outputs'], ['first', 'second', 'third'])

    def test_dry_run_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(tmp) / 'input.json'
            source.write_text(json.dumps([dict(instruction='s', input='p', goal='g')]))
            output = Path(tmp) / 'output.json'
            with patch.object(generate, 'build_model') as build:
                generate.main(['--input', str(source), '--model-path', tmp, '--output', str(output), '--dry-run'])
                build.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_input_and_output_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(tmp) / 'invalid.json'
            source.write_text('[{"input": "missing fields"}]')
            with self.assertRaises(ValueError):
                generate.load_inputs(source)
            output = Path(tmp) / 'results.json'
            with self.assertRaises(ValueError):
                generate.process_file([dict(instruction='s', input='p', goal='g')], output,
                                      Mock(batch_generate_with_system=Mock(return_value=[])), 1)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
