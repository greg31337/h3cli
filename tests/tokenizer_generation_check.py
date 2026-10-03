#!/usr/bin/env python3
"""Verify numerical/structural token-fix render evidence (not perceptual quality)."""
import argparse
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', type=Path, default=ROOT/'outputs/tokenfix-validation/generation')
    args = p.parse_args()
    out = args.directory
    runs = json.loads((out/'results.json').read_text())
    corpus = {c['name']: c for c in json.loads((ROOT/'tests/tokenizer_corpus.json').read_text())}
    names = ['control-before', 'control-after', 'dialogue-before', 'dialogue-after',
             'cutoff-after', 'cutoff-unmarked-after', 'ref-dialogue-before', 'ref-dialogue-after']
    assert all(n in runs for n in names), 'all eight renders must complete'
    assert runs['control-before']['hashes'] == runs['control-after']['hashes'], 'control regression'
    results = {'control_all_artifacts_identical': True, 'pairs': {}}
    for case in ['dialogue', 'ref-dialogue']:
        before, after = runs[case+'-before'], runs[case+'-after']
        prompt_ids = corpus['dialogue-render']['ids']
        assert after['ids'][-len(prompt_ids):] == prompt_ids
        assert not any(151669 <= n <= 151675 for n in before['ids'])
        assert before['hashes']['text'] != after['hashes']['text']
        # All numeric vision markers and reference Qwen/VAE embeddings must be
        # preserved. Prompt length changes; compare axis-major positions as rows.
        preserved = {k: before['hashes'][k] == after['hashes'].get(k)
                     for k in before['hashes'] if k.startswith(('vision-', 'condition-')) or k == 'spans'}
        assert preserved and all(preserved.values()), (case, preserved)
        prefix = len(after['ids'])-len(prompt_ids)
        assert before['ids'][:prefix] == after['ids'][:prefix]
        def positions(name):
            data = (out/(name+'.positions')).read_bytes()
            values = struct.unpack('<'+'I'*(len(data)//4), data)
            n = len(values)//3
            return [values[i*n:i*n+prefix] for i in range(3)]
        assert positions(case+'-before') == positions(case+'-after')
        results['pairs'][case] = {'reference_artifacts_identical': preserved,
                                 'vision_token_prefix_identical': True,
                                 'vision_position_prefix_identical': True,
                                 'corrected_dialogue_ids': prompt_ids}
    cutoff, unmarked = runs['cutoff-after'], runs['cutoff-unmarked-after']
    assert cutoff['ids'] == unmarked['ids']+[151671]
    assert cutoff['hashes']['text'] != unmarked['hashes']['text']
    for key, value in cutoff['hashes'].items():
        if key.startswith(('vision-', 'condition-')) or key == 'spans':
            assert value == unmarked['hashes'][key], ('cutoff reference changed', key)
    results['cutoff_is_one_dedicated_appended_token'] = True
    results['cutoff_reference_artifacts_identical'] = True
    (out/'checks.json').write_text(json.dumps(results, indent=2)+'\n')
    print('ok: identical control AV output, corrected FL2VA/Ref2VA dialogue, unchanged reference embeddings/vision tokens, dedicated cutoff ID')


if __name__ == '__main__':
    main()
