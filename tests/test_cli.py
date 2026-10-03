#!/usr/bin/env python3
"""Check explicit CLI operation selection without loading models or a GPU."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
from pathlib import Path
import subprocess
import tempfile
import unittest


class CLI(unittest.TestCase):
    def run_cli(self, *args, stdin='', cwd=None):
        return subprocess.run([str(Path('bin/h3cli').resolve()), *args], input=stdin, cwd=cwd,
                              capture_output=True, text=True, timeout=10)

    def test_generation_requires_a_command_line_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / 'frames'
            for stdin in ('', 'A fox walks through snow.\n'):
                with self.subTest(stdin=stdin):
                    result = self.run_cli('-d', directory, '--frames-dir', str(frames), stdin=stdin)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('generation requires -p/--prompt', result.stderr)
                    self.assertFalse(frames.exists())

    def test_operations_without_prompts_reach_their_own_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing')
            operations = [
                (['-d', missing, '--info'], missing),
                (['-d', missing, '--resume-sampler-state', missing], 'h3sample: cannot open state'),
                (['-d', missing, '--decode-av-state', missing], 'cannot open h3av state'),
                (['--decode-still-latent', missing, '--image-vae', missing], missing),
            ]
            for args, diagnostic in operations:
                with self.subTest(args=args):
                    result = self.run_cli(*args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertNotIn('generation requires -p/--prompt', result.stderr)

    def test_help_lists_explicit_operations(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('-p PROMPT', result.stderr)
        self.assertIn('--resume-sampler-state PATH', result.stderr)
        self.assertIn('--decode-av-state PATH', result.stderr)
        self.assertIn('--decode-still-latent PATH', result.stderr)
        self.assertNotIn('interactive', result.stderr.lower())
        self.assertIn('default: models/MiniMaxH3', result.stderr)

    def test_default_model_path_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli('--info', cwd=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing required model file', result.stderr)
            self.assertIn('models/MiniMaxH3', result.stderr)
            override = str(Path(directory)/'custom-model')
            result = self.run_cli('--info', '-d', override, cwd=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(override, result.stderr)
            self.assertNotIn('models/MiniMaxH3', result.stderr)


if __name__ == '__main__':
    unittest.main()
