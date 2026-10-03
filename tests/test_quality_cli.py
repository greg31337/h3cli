"""Exercise preset expansion, precedence and operation guards without models."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import platform
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


class QualityCLI(unittest.TestCase):
    binary = str(Path("bin/h3cli").resolve())

    def run_cli(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [self.binary, "-d", directory, *args], cwd=directory,
                capture_output=True, text=True, timeout=20)

    def settings(self, *args):
        # No prompt: resolution is observable, but generation cannot start.
        result = self.run_cli(*args)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        match = re.search(r"^h3cli: quality=(.*)$", result.stderr, re.M)
        self.assertIsNotNone(match, result.stderr)
        return dict(item.split("=", 1) for item in
                    ("quality=" + match.group(1)).split())

    def test_presets(self):
        metal = platform.system() == "Darwin"
        for level, steps, reuse, adaptive, preview in [
            ("lossless", "50", "1", "off", "off"),
            ("extra-high", "50", "1", "off", "off"),
            ("high", "50", "2" if metal else "1",
             "off" if metal else "conservative", "off"),
            ("preview", "12", "2", "off", "on"),
            ("fast-preview", "6", "3", "off", "on"),
        ]:
            with self.subTest(level=level):
                values = self.settings("--quality", level)
                self.assertEqual(values["quality"], level)
                self.assertEqual(values["steps"], steps)
                self.assertEqual(values["reuse"], reuse)
                self.assertEqual(values["adaptive-cache"], adaptive)
                self.assertEqual(values["preview-vae"], preview)
                self.assertEqual(values["layers"], "50")
                self.assertEqual(values["core-reuse"], "1")
                self.assertEqual(values["cuda-attention"], "default")
                self.assertEqual(values["cuda-denoise-quant"], "off")

    def test_explicit_flags_win_in_both_orders(self):
        overrides = ("--steps", "4", "--reuse", "1", "--layers", "45",
                     "--core-reuse", "4", "--adaptive-cache", "off",
                     "--no-preview-vae")
        for level in ("lossless", "extra-high", "high", "preview", "fast-preview"):
            with self.subTest(level=level):
                before = self.settings(*overrides, "--quality", level)
                after = self.settings("--quality", level, *overrides)
                self.assertEqual(before, after)
                for key, value in (("steps", "4"), ("reuse", "1"),
                                   ("layers", "45"), ("core-reuse", "4"),
                                   ("adaptive-cache", "off"), ("preview-vae", "off")):
                    self.assertEqual(after[key], value)

    def test_other_approximations_are_preserved(self):
        overrides = ("--cuda-attention", "subblock", "--cuda-denoise-quant", "fp8",
                     "--adaptive-cache", "aggressive", "--preview-vae")
        for args in ((*overrides, "--quality", "lossless"),
                     ("--quality", "lossless", *overrides)):
            values = self.settings(*args)
            self.assertEqual(values["adaptive-cache"], "aggressive")
            self.assertEqual(values["cuda-attention"], "subblock")
            self.assertEqual(values["cuda-denoise-quant"], "fp8")
            self.assertEqual(values["preview-vae"], "on")

    def test_last_preset_wins_without_leaking_earlier_defaults(self):
        for first, last in (("fast-preview", "lossless"), ("high", "preview"),
                            ("preview", "extra-high")):
            self.assertEqual(self.settings("--quality", first, "--quality", last),
                             self.settings("--quality", last))
        for flags in (("--preview-vae", "--no-preview-vae"),
                      ("--no-preview-vae", "--preview-vae")):
            values = self.settings(*flags, "--quality", "preview")
            self.assertEqual(values["preview-vae"], "off" if flags[-1] == "--no-preview-vae" else "on")

    def test_invalid_levels(self):
        for level in ("", "HIGH", "extra_high", "fast", "50", "high "):
            result = self.run_cli("--quality", level)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--quality must be", result.stderr)
        self.assertEqual(self.run_cli("--quality").returncode, 2)

    def test_authoritative_operations_reject_presets_before_io(self):
        for operation in ("--resume-sampler-state", "--decode-av-state",
                          "--decode-still-latent", "--upscale-state",
                          "--inspect-upscale-state", "--info"):
            flags = (operation,) if operation == "--info" else (operation, "missing")
            for args in ((*flags, "--quality", "high"), ("--quality", "high", *flags)):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--quality requires fresh generation", result.stderr)
                self.assertNotIn("cannot open", result.stderr)

    def test_continuation_and_bridge_accept_expanded_settings(self):
        for mode in ("hard", "bridge"):
            result = self.run_cli("-p", "test", "--continue-from", "missing",
                                  "--continue-mode", mode, "--quality", "lossless",
                                  "--steps", "6")
            self.assertIn("cannot open h3av state", result.stderr)
            self.assertNotIn("--quality requires", result.stderr)

    def test_absent_quality_keeps_existing_path(self):
        result = self.run_cli()
        self.assertIn("generation requires -p/--prompt", result.stderr)
        self.assertNotIn("quality=", result.stderr)

    def test_help(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        for text in ("--quality LEVEL", "lossless", "extra-high", "fast-preview",
                     "--no-preview-vae", "explicit flags win"):
            self.assertIn(text, result.stderr)


if __name__ == "__main__":
    unittest.main()
