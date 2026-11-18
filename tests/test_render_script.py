#!/usr/bin/env python3
"""Native script planning tests; no models or GPU required."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('H3_RENDER_BIN', ROOT / 'bin/h3cli')).resolve()
HEADINGS = ['subject_definitions', 'summary', 'retention_analysis',
            'detailed_description', 'overall_soundscape', 'non_diegetic_music']


class RenderScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3 render's ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'scenes.txt'
        self.out = self.root / 'output'
        self.env = dict(os.environ)
        self.env.pop('H3_TEST_MAX_EVALUATIONS', None)
        self.env.pop('H3_MODEL_DIR', None)
        for name in ('one.png', 'two.png', 'with spaces.png', 'clip.mp4', 'sound.wav'):
            (self.root / name).touch()

    def invoke(self, text, *options, dry=True, binary=BINARY):
        self.source.write_text(text)
        command = [str(binary), '--render', str(self.source), '--output-dir', str(self.out)]
        if dry:
            command += ['--dry-run']
        return subprocess.run(command + list(options), capture_output=True, text=True,
                              env=self.env, cwd=self.root, timeout=20)

    def plan(self, text, *options):
        result = self.invoke(text, *options)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.out.exists())
        return json.loads(result.stdout)

    def command_values(self, row, flag):
        command = row['command']
        return [command[i+1] for i, arg in enumerate(command[:-1]) if arg == '--' + flag]

    def test_seed_set_expands_sequences_without_accumulating_offsets(self):
        plan = self.plan('''#seed-set 100, 200, 0
### first
summary:
First.
### next
#continue
#seed 7
summary:
Next.
### independent
Third.
### reset
#unset seed
Fourth.
''')
        self.assertEqual(plan['seed_set'], [100, 200, 0])
        rows = plan['segments']
        for run, offset in enumerate(plan['seed_set']):
            shots = rows[run*4:run*4+4]
            self.assertEqual([r['seed'] for r in shots], [42+offset, 7+offset, 7+offset, 42+offset])
            self.assertEqual([r['label'] for r in shots], ['first','next','independent','reset'])
            self.assertEqual([r['name'] for r in shots], [f'seed-{offset}/{n}.mp4' for n in ('first','next','independent','reset')])
            for row in shots:
                self.assertEqual(self.command_values(row, 'seed'), [str(row['seed'])])
                self.assertEqual(row['seed_offset'], offset)
            self.assertNotIn('--continue-from', shots[0]['command'])
            self.assertEqual(self.command_values(shots[1], 'continue-from'),
                             self.command_values(shots[0], 'save-av-state'))
            merges = [m for m in plan['merged'] if m['seed_offset']==offset]
            self.assertEqual([m['name'] for m in merges], [f'seed-{offset}/first-merged.mp4',f'seed-{offset}/scenes.mp4'])
            self.assertEqual(merges[-1]['segments'], [r['name'] for r in shots])
        self.assertEqual(len({self.command_values(r,'save-av-state')[0] for r in rows}), 12)

    def test_seed_set_cli_default_and_uint64_boundary(self):
        plan = self.plan('#seed-set=1,2\n### a\nOne.\n### b\nTwo.', '--seed', '100')
        self.assertEqual([s['seed'] for s in plan['segments']], [101,101,102,102])
        plan = self.plan('#seed-set 18446744073709551615\n### a\n#seed 0\nOne.')
        self.assertEqual(plan['segments'][0]['seed'], 2**64-1)
        self.assertEqual(self.command_values(plan['segments'][0], 'seed'), [str(2**64-1)])

    def test_seed_set_rejects_malformed_duplicates_and_overflow(self):
        for values in ('', '1,', ',1', '1,,2', '-1', '+1', '1.5', '1 2', '1,01',
                       '18446744073709551616', '1e2'):
            with self.subTest(values=values):
                result = self.invoke(f'#seed-set {values}\n### a\nOne.')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('#seed-set', result.stderr)
                self.assertFalse(self.out.exists())
        for text in ('#seed-set 1\n#seed-set 2\n### a\nOne.',
                     '#seed-set 1\n### a\n#seed 18446744073709551615\nOne.',
                     '#seed-set 18446744073709551615\n### a\nOne.'):
            result = self.invoke(text)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('seed', result.stderr)
            self.assertFalse(self.out.exists())

    def test_prelude_directives_must_precede_even_skipped_segments(self):
        for directive in ('#seed-set 1,2', '#overlay-segment-name', '#ref actor one.png'):
            for heading in ('### a', '!### a'):
                result = self.invoke(f'{heading}\n{directive}\nOne.')
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('before the first segment', result.stderr)
        result = self.invoke('#overlay-segment-name true\n### a\nOne.')
        self.assertEqual(result.returncode, 2)
        self.assertIn('takes no arguments', result.stderr)

    def test_overlay_names_and_literal_directives(self):
        plan = self.plan('#overlay-segment-name\n#overlay-segment-name\n### opening shot.mp4\nOne.')
        self.assertTrue(plan['overlay_segment_name'])
        self.assertEqual(plan['segments'][0]['label'], 'opening shot.mp4')
        self.assertNotIn('--overlay-segment-name', plan['segments'][0]['command'])
        plan = self.plan('## #overlay-segment-name\n## #seed-set 1,2\n### a\n'
                         '\\#overlay-segment-name\n\\#seed-set 1,2\n\\#ref actor one.png')
        self.assertFalse(plan['overlay_segment_name'])
        self.assertEqual(plan['seed_set'], [])
        self.assertEqual(plan['references'], {})
        self.assertIn('#ref actor one.png', plan['segments'][0]['prompt'])

    def test_reference_aliases_all_media_and_literal_escape(self):
        plan = self.plan('''#ref actor "with spaces.png"
#ref movie clip.mp4
#ref voice sound.wav
#ref one.png two.png
### a
#refimage actor
#ref-image one.png
#ref-image ./one.png
#ref-video movie
#ref-silent-video movie
#refaudio voice
#ref-video-audio movie voice
One.
### b
Two.
''')
        row = plan['segments'][0]
        self.assertEqual(plan['references']['actor'], str(self.root/'with spaces.png'))
        self.assertEqual(self.command_values(row, 'ref-image'),
                         [str(self.root/n) for n in ('with spaces.png','two.png','one.png')])
        for flag, name in (('ref-video','clip.mp4'), ('ref-silent-video','clip.mp4'), ('ref-audio','sound.wav')):
            self.assertEqual(self.command_values(row, flag), [str(self.root/name)])
        at = row['command'].index('--ref-video-audio')
        self.assertEqual(row['command'][at+1:at+3], [str(self.root/'clip.mp4'), str(self.root/'sound.wav')])
        self.assertNotIn('--ref-image', plan['segments'][1]['command'])
        anchors = self.plan('#ref start one.png\n#ref end two.png\n### a\n#first-frame start\n#last-frame end\nOne.')
        self.assertEqual(self.command_values(anchors['segments'][0], 'first-frame'), [str(self.root/'one.png')])

    def test_reference_aliases_lazy_validation_and_invalid_definitions(self):
        plan = self.plan('#ref unused missing.png\n!### skipped\n#ref-image unused\n### a\nOne.')
        self.assertEqual(plan['segments'][0]['status'], 'skipped')
        for definition in ('#ref', '#ref actor', '#ref actor one.png extra', '#ref "" one.png',
                           '#ref actor ""', '#ref ../actor one.png', '#ref "two names" one.png',
                           '#ref actor one.png\n#ref actor two.png'):
            result = self.invoke(definition+'\n### a\nOne.')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('#ref', result.stderr)
        result = self.invoke('#ref missing missing.png\n### a\n#ref-image missing\nOne.')
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(self.root/'missing.png'), result.stderr)
        result = self.invoke('### a\n#ref-image undefined_alias\nOne.')
        self.assertEqual(result.returncode, 2)
        self.assertIn('undefined_alias', result.stderr)

    def test_seed_set_nomerge_and_all_skipped_remain_noops(self):
        text = '#seed-set 1,2\n#overlay-segment-name\n#ref absent missing.png\n!### a\n#ref-image absent\n#nomerge'
        plan = self.plan(text)
        self.assertEqual(plan['merged'], [])
        self.assertEqual([r['status'] for r in plan['segments']], ['skipped','skipped'])
        result = self.invoke(text, dry=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.out.exists())

    def test_help_and_mode_conflicts(self):
        result = subprocess.run([str(BINARY), '--help'], capture_output=True, text=True)
        for flag in ('--render', '--output-dir', '--dry-run', '#nomerge', '#overlay-segment-name', '#seed-set', '#ref ALIAS'):
            self.assertIn(flag, result.stdout + result.stderr)
        for flags in (['--dry-run'], ['--output-dir', 'out'],
                      ['--server', '--render', 'missing'], ['--render', 'missing', '--server']):
            result = subprocess.run([str(BINARY), *flags], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_script_wide_and_chain_names_default_directory(self):
        self.source.write_text('### first\nOne.\n### next\n#continue\nTwo.\n### independent\nThree.')
        result = subprocess.run([str(BINARY), '--render', str(self.source), '--dry-run'],
                                capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(Path(plan['output']), self.root / 'scenes-render')
        self.assertEqual([(m['name'], m['segments']) for m in plan['merged']], [
            ('first-merged.mp4', ['first.mp4', 'next.mp4']),
            ('scenes.mp4', ['first.mp4', 'next.mp4', 'independent.mp4'])])
        self.assertEqual([s['delivered'] for s in plan['segments']], [56, 17, 56])
        self.assertFalse((self.root / 'scenes-render').exists())

    def test_inheritance_and_reference_reset(self):
        plan = self.plan('''### 1
#frames 243
#size 1024x768
#steps 30
#seed 18446744073709551615
#refimage one.png
First.
### 2.mp4
#continue
#refimage two.png
Second.
### 3
Independent.
### 4
#continue
Fourth.
''')
        rows = plan['segments']
        self.assertEqual([s['delivered'] for s in rows], [243, 204, 243, 204])
        for row in rows:
            self.assertEqual(self.command_values(row, 'steps'), ['30'])
            self.assertEqual(self.command_values(row, 'seed'), ['18446744073709551615'])
        self.assertEqual(self.command_values(rows[0], 'ref-image'), [str(self.root / 'one.png')])
        self.assertEqual(self.command_values(rows[1], 'ref-image'), [str(self.root / 'two.png')])
        self.assertFalse(self.command_values(rows[2], 'ref-image'))

    def test_all_reference_types_order_anchors_and_paths(self):
        plan = self.plan('''### first
#refimage one.png
#ref-audio sound.wav
#ref-image two.png
#ref-video-audio clip.mp4 sound.wav
#ref-image-size high
One.
### next
Two.
''')
        command = plan['segments'][0]['command']
        media = [s for s in command if s in ('--ref-image', '--ref-audio', '--ref-video-audio', '--first-frame', '--last-frame')]
        self.assertEqual(media, ['--ref-image', '--ref-audio', '--ref-image', '--ref-video-audio'])
        self.assertEqual(self.command_values(plan['segments'][1], 'ref-image-size'), ['high'])
        self.assertNotIn('--first-frame', plan['segments'][1]['command'])
        anchors = self.plan('### a\n#first-frame one.png\n#last-frame two.png\nOne.\n### b\nTwo.')['segments']
        self.assertIn('--first-frame', anchors[0]['command'])
        self.assertNotIn('--first-frame', anchors[1]['command'])

    def test_cli_defaults_script_override_and_unset(self):
        plan = self.plan('### a\nOne.\n### b\n#steps 4\n#unset seed\nTwo.', '--steps', '2', '--seed', '9')
        self.assertEqual(self.command_values(plan['segments'][0], 'steps'), ['2'])
        self.assertEqual(self.command_values(plan['segments'][1], 'steps'), ['4'])
        self.assertFalse(self.command_values(plan['segments'][1], 'seed'))

    def test_lora_replacement_and_unset(self):
        plan = self.plan('''### a
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
''')
        rows = plan['segments']
        self.assertEqual(self.command_values(rows[0], 'lora'), self.command_values(rows[1], 'lora'))
        self.assertEqual(self.command_values(rows[2], 'lora'), [str(self.root/'c.safetensors')+':0', str(self.root/'d.safetensors')+':1.2'])
        self.assertNotIn('--profile', rows[2]['command'])
        self.assertFalse(self.command_values(rows[3], 'lora'))

    def test_exclusive_settings_and_native_frame_alignment(self):
        plan = self.plan('''### a
#frames 243
#preview-vae
#ffmpeg-crf 8
One.
### b
#seconds 0.9375
#no-preview-vae
#output-quality high
Two.
### c
#frames 40
#lossless-video
Three.
''')
        rows = plan['segments']
        self.assertEqual([s['frames'] for s in rows], [243, 39, 56])
        for flag in ('--frames', '--preview-vae', '--ffmpeg-crf'):
            self.assertNotIn(flag, rows[1]['command'])
        for flag in ('--seconds', '--output-quality'):
            self.assertNotIn(flag, rows[2]['command'])

    def test_quotes_bom_comments_and_multiline_prompt(self):
        plan = self.plan('\ufeff## Example\r\n### shot one ###\r\n#--frames=56 # note\r\n'
                         '#refimage "with spaces.png"\r\n#--model-dir "model dir"\r\n'
                         'First line.\r\n\r\n\\# A literal hash.\r\nSecond line.\r\n')
        row = plan['segments'][0]
        self.assertEqual(row['name'], 'shot one.mp4')
        self.assertEqual(row['prompt'], 'First line.\n\n# A literal hash.\nSecond line.')
        self.assertEqual(self.command_values(row, 'ref-image'), [str(self.root/'with spaces.png')])
        self.assertEqual(self.command_values(row, 'model-dir'), [str(self.root/'model dir')])

    def test_utf8_and_oversized_dimensions(self):
        row = self.plan('### 猫\n#size 256X256\nA cat 猫 and a tree 🌲.')['segments'][0]
        self.assertEqual(row['prompt'], 'A cat 猫 and a tree 🌲.')
        result = self.invoke('### a\n#size 999999999999999999999999999x256\nText.')
        self.assertEqual(result.returncode, 2)
        for invalid in (b'\xff', b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xe2\x82', b'\0'):
            self.source.write_bytes(b'### a\nText '+invalid)
            result = subprocess.run([str(BINARY), '--render', str(self.source), '--dry-run'], capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_bridge_only_applies_to_continuations_and_full_state_context(self):
        rows = self.plan('### a\n#frames 90\n#continue-mode bridge\n#continue-context 39\n'
                         '#continue-bridge-profile linear\nOne.\n### b\n#continue\nTwo.\n'
                         '### c\n#continue\nThree.\n### d\nFour.')['segments']
        self.assertEqual([s['delivered'] for s in rows], [90, 51, 51, 90])
        for row in (rows[0], rows[3]):
            self.assertNotIn('--continue-mode', row['command'])
        self.assertEqual(self.command_values(rows[1], 'continue-mode'), ['bridge'])
        self.assertTrue(self.command_values(rows[2], 'continue-from')[0].endswith('/0001.h3av'))

    def test_each_section_inherits_and_can_be_replaced_or_cleared(self):
        for heading in HEADINGS:
            with self.subTest(heading=heading):
                rows = self.plan(f'### a\n{heading}:\nOld.\n### b\n#continue\n'
                                 f'### c\n#continue\n{heading}:\nNew.\n'
                                 f'### d\n#continue\nPreamble.\n{heading}:\n'
                                 '### e\n#continue\nAnother preamble.')['segments']
                self.assertEqual(rows[0]['prompt'], f'{heading}:\nOld.')
                self.assertEqual(rows[1]['prompt'], rows[0]['prompt'])
                self.assertEqual(rows[2]['prompt'], f'{heading}:\nNew.')
                self.assertEqual(rows[3]['prompt'], 'Preamble.')
                self.assertEqual(rows[4]['prompt'], 'Another preamble.')

    def test_canonical_order_partial_overrides_transitive_and_reset(self):
        original = '\n'.join(f'{h}:\n{h} body\n  indented detail' for h in reversed(HEADINGS))
        rows = self.plan(f'### a\nOriginal preamble.\n{original}\n'
                         '### b\n#continue\nNew preamble.\nsummary:\nReplaced.\n'
                         '### c\n#continue\ndetailed_description:\nNew detail.\n'
                         '### d\nIndependent.\n### e\n#continue\nStill independent.')['segments']
        for row in rows[:3]:
            self.assertEqual([line[:-1] for line in row['prompt'].splitlines() if line[:-1] in HEADINGS], HEADINGS)
        self.assertIn('summary:\nReplaced.', rows[2]['prompt'])
        self.assertNotIn('New preamble.', rows[2]['prompt'])
        self.assertIn('  indented detail', rows[2]['prompt'])
        self.assertEqual(rows[3]['prompt'], 'Independent.')
        self.assertEqual(rows[4]['prompt'], 'Still independent.')
        self.assertEqual(self.command_values(rows[2], 'prompt'), [rows[2]['prompt']])

    def test_plain_preamble_unknown_headings_and_all_overrides(self):
        rows = self.plan('### a\nsummary:\nOld.\n### b\n#continue\n'
                         'Preamble.\nsummary:\nNew.\nunknown_heading:\nBody.')['segments']
        self.assertEqual(rows[1]['prompt'], 'Preamble.\n\nsummary:\nNew.\nunknown_heading:\nBody.')
        original = '\n'.join(f'{h}:\nOld {h}.' for h in HEADINGS)
        replacement = '\n'.join(f'{h}:\nNew {h}.' for h in HEADINGS)
        rows = self.plan(f'### a\n{original}\n### b\n#continue\n{replacement}')['segments']
        self.assertNotIn('Old', rows[1]['prompt'])
        self.assertEqual(rows[1]['prompt'].count('New'), 6)

    def test_duplicate_headings_and_empty_effective_prompt(self):
        for text, diagnostic in [('### a\nsummary:\nOne.\nsummary:\nTwo.', ':4: duplicate'),
                                 ('### a\nsummary:\nOne.\n### b\n#continue\nsummary:', 'empty prompt')]:
            result = self.invoke(text)
            self.assertEqual(result.returncode, 2)
            self.assertIn(diagnostic, result.stderr)

    def test_nomerge_in_every_location_is_global_and_idempotent(self):
        base = '### a\nOne.\n### b\n#continue\nTwo.\n!### skipped\n'
        variants = ['#nomerge\n' + base, base + '#nomerge\n',
                    base.replace('One.', '#nomerge\nOne.'), base.replace('Two.', 'Two.\n#nomerge'),
                    '#nomerge\n' + base + '#nomerge\n']
        for text in variants:
            plan = self.plan(text)
            self.assertTrue(plan['nomerge'])
            self.assertEqual(plan['merged'], [])
            self.assertEqual([s['prompt'] for s in plan['segments'][:2]], ['One.', 'Two.'])

    def test_nomerge_comments_literals_and_arguments(self):
        plan = self.plan('## #nomerge\n### a\nUse #nomerge in prose.\n\\#nomerge')
        self.assertFalse(plan['nomerge'])
        self.assertEqual(len(plan['merged']), 1)
        self.assertIn('\n#nomerge', plan['segments'][0]['prompt'])
        result = self.invoke('### a\nText.\n#nomerge true')
        self.assertEqual(result.returncode, 2)
        self.assertIn('no arguments', result.stderr)

    def test_skip_directives_inherit_but_prompts_and_references_do_not(self):
        rows = self.plan('!### a\n#steps 4\n#refimage missing.png\nsummary:\nSkipped.\n'
                         '!### b\n#continue\n### c\nActive.\n### d\n#continue\nNext.')['segments']
        self.assertEqual([s['status'] for s in rows], ['skipped', 'skipped', 'planned', 'planned'])
        self.assertEqual(rows[2]['prompt'], 'Active.')
        self.assertEqual(self.command_values(rows[2], 'steps'), ['4'])
        self.assertNotIn('--ref-image', rows[2]['command'])

    def test_all_skipped_no_outputs_or_media_dependencies(self):
        self.env.update(H3_FFMPEG='/missing/ffmpeg', H3_FFPROBE='/missing/ffprobe')
        result = self.invoke('!### a\n#refimage missing.png\n!### b\n#continue', dry=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.out.exists())

    def test_late_errors_prevent_any_rendering_or_outputs(self):
        cases = {
            'first continuation': ('### a\n#continue\nOne.', 'preceding'),
            'mixed skipped': ('### good\nGood.\n### a\nOne.\n!### b\n#continue\nTwo.', 'mixes'),
            'duplicate': ('### a\nOne.\n### a\nTwo.', 'collides'),
            'case collision': ('### A\nOne.\n### a\nTwo.', 'collides'),
            'chain collision': ('### a\nOne.\n### b\n#continue\nTwo.\n### a-merged\nThree.', 'collides'),
            'script collision': ('### scenes\nOne.', 'collides'),
            'unknown': ('### a\n#frmes 56\nOne.', 'unknown'),
            'empty': ('### a\n#frames 56', 'empty prompt'),
            'bad quote': ('### a\n#refimage "one.png\nOne.', 'quotation'),
            'bad size': ('### a\n#size 1023x768\nOne.', 'multiples'),
            'changed size': ('### a\nOne.\n### b\n#continue\n#size 512x512\nTwo.', 'width'),
            'missing ref': ('### a\n#refimage missing.png\nOne.', 'missing'),
            'mode switch': ('### a\n#refimage one.png\nOne.\n### b\n#continue\nTwo.', 'Ref2VA'),
            'short source': ('### a\n#frames 22\nOne.\n### b\n#frames 56\n#continue\nTwo.', 'context'),
            'no suffix': ('### a\n#frames 39\nOne.\n### b\n#continue\nTwo.', 'context'),
            'bad context': ('### a\nOne.\n### b\n#continue\n#continue-context 40\nTwo.', 'context'),
            'bad path': ('### ../a\nOne.', 'filename'),
            'preamble': ('unexpected\n### a\nOne.', 'first video'),
            'late option': ('### a\nOne.\n#frames 56', 'before the prompt'),
            'bad unset': ('### a\n#unset frmes\nOne.', 'cannot unset'),
            'zero seconds': ('### a\n#seconds 0\nOne.', 'seconds'),
            'nan': ('### a\n#seconds nan\nOne.', 'invalid'),
            'overflow': ('### a\n#seed 18446744073709551616\nOne.', 'invalid'),
            'missing value': ('### a\n#frames\nOne.', 'expects 1'),
            'flag value': ('### a\n#profile true\nOne.', 'expects 0'),
            'bad enum': ('### a\n#continue-mode wrong\nOne.', 'hard|bridge'),
        }
        for label, (text, diagnostic) in cases.items():
            with self.subTest(label=label):
                result = self.invoke(text, dry=False)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(self.out.exists())

    def test_disabled_merge_names_do_not_collide(self):
        plan = self.plan('### scenes\nOne.\n### b\n#continue\nTwo.\n### scenes-merged\nThree.\n#nomerge')
        self.assertEqual(plan['merged'], [])

    def test_managed_and_nonvideo_modes_rejected(self):
        flags = {'prompt': 'Text', 'output': 'out.mp4', 'continue-from': 'state.h3av',
                 'save-av-state': 'state.h3av', 'keep-continuation-prefix': None,
                 'still': None, 'info': None, 'state-only': None, 'stop-after-step': '1',
                 'resume-sampler-state': 'state.h3sp', 'download-models': 'all', 'list-models': None}
        for name, value in flags.items():
            with self.subTest(name=name):
                options = ['--' + name] + ([value] if value is not None else [])
                result = self.invoke('### a\nOne.', *options)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('managed', result.stderr)

    def test_native_combination_validation_runs_before_outputs(self):
        result = self.invoke('### a\n#lora-cache cache\nText.', dry=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('require --lora', result.stderr)
        self.assertFalse(self.out.exists())

if __name__ == '__main__':
    unittest.main()
