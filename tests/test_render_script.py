#!/usr/bin/env python3
"""Parser and execution-plan checks; no models, GPU or media tools required."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render


class RenderScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = render.option_inventory()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h3 render's ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.input = self.root / "scenes.txt"
        for name in ("one.png", "two.png", "with spaces.png", "clip.mp4", "sound.wav"):
            (self.root / name).touch()

    def parse(self, text, validate=True):
        self.input.write_text(text)
        result = render.read_sections(self.input, self.options)
        if validate:
            render.validate_plan(result, self.options)
        return result

    def test_inheritance_reference_reset_and_chain_boundaries(self):
        sections = self.parse("""### 1
#frames 243
#size 1024x768
#steps 30
#seed 76
#refimage one.png
First prompt.
### 2.mp4
#continue
#refimage two.png
Second prompt.
### 3
Independent prompt.
### 4
#continue
Fourth prompt.
""")
        self.assertEqual(render.chains(sections), [[0, 1], [2, 3]])
        self.assertEqual([s.delivered for s in sections], [243, 204, 243, 204])
        self.assertEqual([s.value("seed", self.options) for s in sections], ["76"] * 4)
        self.assertEqual([s.value("steps", self.options) for s in sections], ["30"] * 4)
        self.assertEqual([len(s.references) for s in sections], [1, 1, 0, 0])
        self.assertEqual(sections[0].references[0][1], [str(self.root / "one.png")])
        self.assertEqual(sections[1].references[0][1], [str(self.root / "two.png")])

    def test_reference_order_all_media_and_anchors_reset(self):
        sections = self.parse("""### first
#refimage one.png
#ref-audio sound.wav
#ref-image two.png
#ref-video-audio clip.mp4 sound.wav
#first-frame one.png
#last-frame two.png
#ref-image-size high
One.
### next
Two.
""")
        self.assertEqual([name for name, _ in sections[0].references],
                         ["ref-image", "ref-audio", "ref-image", "ref-video-audio", "first-frame", "last-frame"])
        self.assertEqual(sections[1].references, [])
        self.assertEqual(sections[1].value("ref-image-size", self.options), "high")

    def test_lora_list_replacement_and_unset(self):
        sections = self.parse("""### a
#lora a.safetensors:0.8
#lora b.safetensors
#profile
One.
### b
Two.
### c
#lora c.safetensors:0
#lora d.safetensors:1.2
#unset profile
Three.
### d
#unset lora
Four.
""")
        self.assertEqual(sections[0].settings["lora"], sections[1].settings["lora"])
        self.assertEqual(sections[2].settings["lora"],
                         [[str(self.root / "c.safetensors") + ":0"], [str(self.root / "d.safetensors") + ":1.2"]])
        self.assertNotIn("profile", sections[2].settings)
        self.assertNotIn("lora", sections[3].settings)

    def test_mutually_exclusive_settings_replace_inherited_values(self):
        sections = self.parse("""### a
#frames 243
#preview-vae
#ffmpeg-crf 8
One.
### b
#seconds 1
#no-preview-vae
#output-quality high
Two.
### c
#frames 22
#lossless-video
Three.
""")
        self.assertEqual([s.frames for s in sections], [243, 39, 22])
        self.assertNotIn("frames", sections[1].settings)
        self.assertNotIn("seconds", sections[2].settings)
        self.assertNotIn("preview-vae", sections[1].settings)
        self.assertNotIn("ffmpeg-crf", sections[1].settings)
        self.assertNotIn("output-quality", sections[2].settings)

    def test_quotes_bom_comments_and_multiline_prompt(self):
        sections = self.parse('\ufeff## Example\n### shot one ###\n#--frames=56 # note\n'
                              '#refimage "with spaces.png"\n#--model-dir "model dir"\n'
                              'First line.\n\n\\# A literal hash.\nSecond line.\n')
        self.assertEqual(sections[0].name, "shot one.mp4")
        self.assertEqual(sections[0].prompt, "First line.\n\n# A literal hash.\nSecond line.")
        self.assertEqual(sections[0].references[0][1], [str(self.root / "with spaces.png")])
        self.assertEqual(sections[0].settings["model-dir"], [[str(self.root / "model dir")]])

    def test_native_seconds_rounding_and_frame_alignment(self):
        # C llround rounds .5 upward, unlike Python round's ties-to-even rule.
        sections = self.parse("### a\n#seconds 0.9375\nOne.\n### b\n#frames 40\nTwo.\n")
        self.assertEqual([s.frames for s in sections], [39, 56])

    def test_context_uses_full_previous_state_not_trimmed_clip(self):
        sections = self.parse("### a\n#frames 90\nOne.\n### b\n#continue\nTwo.\n"
                              "### c\n#continue\nThree.\n")
        self.assertEqual([s.delivered for s in sections], [90, 51, 51])

    def test_bridge_settings_apply_only_to_continuations(self):
        sections = self.parse("### a\n#continue-mode bridge\n#continue-context 39\n"
                              "#continue-bridge-profile linear\nOne.\n### b\n#continue\nTwo.\n"
                              "### c\nThree.\n")
        work = self.root / "work"
        commands = [render.render_command("h3cli", s, i, work) for i, s in enumerate(sections)]
        self.assertNotIn("--continue-mode", commands[0])
        self.assertNotIn("--continue-context", commands[0])
        self.assertNotIn("--continue-bridge-profile", commands[2])
        self.assertIn("bridge", commands[1])
        index = commands[1].index("--continue-from")
        self.assertEqual(commands[1][index + 1], str(work / "0000.h3av"))
        self.assertNotIn("--continue-from", commands[2])

    def test_prompt_and_paths_remain_single_arguments(self):
        sections = self.parse("### a\n#refimage 'with spaces.png'\nA \"quoted\" $(touch example) prompt.\n")
        command = render.render_command("h3cli", sections[0], 0, self.root)
        self.assertEqual(command[command.index("--prompt") + 1], sections[0].prompt)
        self.assertEqual(command[command.index("--ref-image") + 1], str(self.root / "with spaces.png"))

    def test_multiple_chains_and_singletons_have_separate_merges(self):
        sections = self.parse("### a\nOne.\n### b\n#continue\nTwo.\n### c\nThree.\n"
                              "### d\nFour.\n### e\n#continue\nFive.\n")
        self.assertEqual(render.chains(sections), [[0, 1], [2], [3, 4]])
        self.assertEqual([render.merged_name(sections[g[0]]) for g in render.chains(sections) if len(g) > 1],
                         ["a-merged.mp4", "d-merged.mp4"])

    def test_invalid_inputs_fail_before_any_render(self):
        cases = {
            "first continuation": ("### a\n#continue\nOne.", "preceding"),
            "duplicate name": ("### a\nOne.\n### a.mp4\nTwo.", "duplicate"),
            "merge collision": ("### a\nOne.\n### b\n#continue\nTwo.\n### a-merged\nThree.", "collides"),
            "unknown": ("### a\n#frmes 56\nOne.", "Unknown"),
            "empty": ("### a\n#frames 56", "empty prompt"),
            "malformed quote": ('### a\n#refimage "one.png\nOne.', "quotation"),
            "bad size": ("### a\n#size 1023x768\nOne.", "multiples"),
            "changed geometry": ("### a\nOne.\n### b\n#continue\n#size 512x512\nTwo.", "width"),
            "missing reference": ("### a\n#refimage missing.png\nOne.", "missing"),
            "model switch": ("### a\n#refimage one.png\nOne.\n### b\n#continue\nTwo.", "Ref2VA"),
            "source too short": ("### a\n#frames 22\nOne.\n### b\n#frames 56\n#continue\nTwo.", "context"),
            "no suffix": ("### a\n#frames 39\nOne.\n### b\n#continue\nTwo.", "context"),
            "invalid context": ("### a\nOne.\n### b\n#continue\n#continue-context 40\nTwo.", "context"),
            "unsafe path": ("### ../a\nOne.", "filename"),
            "preamble": ("unexpected text\n### a\nOne.", "first video"),
            "late directive": ("### a\nOne.\n#frames 56", "before the prompt"),
            "unknown unset": ("### a\n#unset frmes\nOne.", "Cannot unset"),
            "zero seconds": ("### a\n#seconds 0\nOne.", "seconds"),
            "nan": ("### a\n#seconds nan\nOne.", "Invalid"),
            "seed overflow": ("### a\n#seed 18446744073709551616\nOne.", "Invalid"),
            "no option value": ("### a\n#frames\nOne.", "expects 1"),
            "flag with value": ("### a\n#profile true\nOne.", "expects 0"),
            "invalid enum": ("### a\n#continue-mode wrong\nOne.", "hard|bridge"),
        }
        for name, (text, error) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                self.parse(text)

    def test_script_managed_and_nonvideo_options_are_rejected(self):
        for name in render.MANAGED | render.NON_VIDEO:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "managed|complete video"):
                self.parse(f"### a\n#{name}\nOne.")

    def test_dry_run_has_no_outputs_and_does_not_need_a_binary(self):
        self.input.write_text("### a\nOne.\n### b\n#continue\nTwo.")
        out = self.root / "output"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = render.main([str(self.input), "--dry-run", "--no-env", "--h3cli", "/missing/binary",
                                "-o", str(out)])
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())
        self.assertIn("a-merged.mp4", stdout.getvalue())
        self.assertIn("--continue-from", stdout.getvalue())

    def test_interrupt_stops_render_process_group_and_records_interruption(self):
        # An actual subprocess tree verifies Ctrl-C also reaches h3cli's workers.
        fake = self.root / "fake-h3cli"
        worker = self.root / "worker.py"
        pidfile = self.root / "ready.json"
        worker.write_text("""import os, signal, sys, time
from pathlib import Path
def stopped(signum, frame):
    Path(sys.argv[1]).write_text('stopped')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
Path(sys.argv[2]).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
""")
        fake.write_text(f"""#!{sys.executable}
import json, os, signal, subprocess, sys, time
from pathlib import Path
if '--help' in sys.argv:
    raise SystemExit(0)
root = Path({str(self.root)!r})
def stopped(signum, frame):
    child.wait(timeout=5)
    (root/'parent-stopped').write_text('stopped')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
child = subprocess.Popen([sys.executable, str(root/'worker.py'), str(root/'worker-stopped'), str(root/'worker-ready')])
while not (root/'worker-ready').exists():
    time.sleep(0.01)
(root/'ready.json').write_text(json.dumps([os.getpid(), child.pid]))
while True:
    time.sleep(0.05)
""")
        fake.chmod(0o755)
        self.input.write_text("### a\nOne.")
        out = self.root / "interrupted"
        child = subprocess.Popen([sys.executable, str(render.ROOT / "render.py"), str(self.input),
                                  "--no-env", "--h3cli", str(fake), "--ffmpeg", "/usr/bin/true",
                                  "--ffprobe", "/usr/bin/true", "-o", str(out)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pidfile.exists(), "fake render did not start")
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "running")
            child.send_signal(signal.SIGINT)
            output, _ = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 130, output.decode())
            self.assertTrue((self.root / "parent-stopped").exists())
            self.assertTrue((self.root / "worker-stopped").exists())
            report = json.loads((out / "run.json").read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["segments"][0]["status"], "interrupted")
            self.assertFalse((out / "a.mp4").exists())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
            if pidfile.exists():
                for pid in json.loads(pidfile.read_text()):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
