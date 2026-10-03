"""Acceptance must reject missing data, timing overlap and candidate switching."""
import json
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from metal_fp16_validate import performance_pass, timing_gate
from metal_fp16_model_validate import performance_gate, region_gate

LIMITS = json.loads((Path(__file__).parent/'metal_fp16_limits_v2.json').read_text())['performance']
MODEL_LIMITS = json.loads((Path(__file__).parent/'metal_fp16_model_limits.json').read_text())


class Records(unittest.TestCase):
    def record(self, a=0.8, b=0.9):
        return {'balanced_seconds': {'mpsgraph': [1.0]*6,
                                     'fp16-32x16': [a]*6, 'fp16-64x32': [b]*6}}

    def test_noise_is_not_a_win(self):
        record = self.record()
        record['balanced_seconds']['fp16-32x16'][0] = 1.1
        self.assertFalse(timing_gate(record, LIMITS)['fp16-32x16']['pass'])

    def test_truncated_and_nonfinite(self):
        for bad in ([.8]*5, [.8]*5+[float('nan')], [.8]*5+[0]):
            record = self.record()
            record['balanced_seconds']['fp16-32x16'] = bad
            with self.assertRaises(ValueError):
                timing_gate(record, LIMITS)

    def test_complete_fixed_candidate(self):
        values = {name: timing_gate(self.record(), LIMITS) for name in LIMITS['required_fixtures']}
        self.assertTrue(performance_pass(values, LIMITS))
        del values['block49']
        self.assertFalse(performance_pass(values, LIMITS))

    def test_no_per_fixture_cherry_picking(self):
        values = {name: timing_gate(self.record(.8, 1.1), LIMITS) for name in LIMITS['required_fixtures']}
        values['block49'] = timing_gate(self.record(1.1, .8), LIMITS)
        self.assertFalse(performance_pass(values, LIMITS))

    def model_records(self):
        result = {}
        for name in ('reference-B1', 'fp16-B1', 'reference-B5', 'fp16-B5'):
            n = 1 if name.endswith('B1') else 5
            result[name] = {'evaluation_contract_pass': True, 'geometry': [640,480,243],
                'evaluations': n, 'blocks': 50, 'weight_precision': 'bf16',
                'manifest': {'environment': {'H3_DIT_COMMAND_BLOCKS': '5'}},
                'steps': [{'step': s+1, 'wall_seconds': 100 if name.startswith('reference') else 85,
                           'metal_peak_bytes': 1<<35} for s in range(n)]}
        return result

    def test_model_gate_recomputes_timings(self):
        r = self.model_records()
        self.assertTrue(performance_gate(r, MODEL_LIMITS)['pass'])
        r['fp16-B5']['timing'] = {'steady_median': 1}  # forged cached summary
        for s in r['fp16-B5']['steps']:
            s['wall_seconds'] = 98
        self.assertFalse(performance_gate(r, MODEL_LIMITS)['pass'])

    def test_model_rejects_incomplete_or_diagnostic(self):
        for key, value in [('capture_steps', True), ('capture_ranges', True), ('teacher_from', '/reference'),
                           ('blocks', 5), ('geometry', [640,480,362])]:
            r = self.model_records()
            r['fp16-B5'][key] = value
            with self.assertRaises(ValueError):
                performance_gate(r, MODEL_LIMITS)
        r = self.model_records()
        r['fp16-B5']['steps'].pop()
        with self.assertRaises(ValueError):
            performance_gate(r, MODEL_LIMITS)

    def test_model_memory_and_variability(self):
        r = self.model_records()
        r['fp16-B5']['steps'][-1]['wall_seconds'] = 105
        self.assertFalse(performance_gate(r, MODEL_LIMITS)['pass'])
        r = self.model_records()
        r['fp16-B1']['steps'][0]['metal_peak_bytes'] = 1<<38
        self.assertFalse(performance_gate(r, MODEL_LIMITS)['pass'])

    def test_region_gate_requires_all_warm_blocks(self):
        r = {'reference-profile': {'evaluation_contract_pass': True, 'region_fences': True,
             'components': [{'name': name, 'block': b, 'wall_seconds': 1.0} for b in range(50)
                            for name in ('DiT complete block', 'DiT complete attention region')]}}
        r['fp16-profile'] = copy.deepcopy(r['reference-profile'])
        self.assertTrue(region_gate(r)['no_warm_block_regression'])
        r['fp16-profile']['components'][-1]['wall_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            region_gate(r)
        r['fp16-profile']['components'].pop()
        with self.assertRaises(ValueError):
            region_gate(r)


if __name__ == '__main__':
    unittest.main()
