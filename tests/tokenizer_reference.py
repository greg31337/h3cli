#!/usr/bin/env python3
"""Print released Hugging Face token IDs; optionally check native/baseline parity.

Reference mode needs transformers and uses local files only. --golden-only checks
committed official IDs without Python packages or model downloads.
"""
import argparse
import json
from pathlib import Path
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
H3 = ['<d>', '</d>', '<|cutoff|>', '<|lyrics_start|>', '<|lyrics_end|>',
      '<|caption_start|>', '<|caption_end|>']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tokenizer', nargs='?', type=Path, default=ROOT/'models/MiniMax-H3/FL2VA/tokenizer')
    p.add_argument('--corpus', type=Path, default=ROOT/'tests/tokenizer_corpus.json')
    p.add_argument('--native', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--golden-only', action='store_true')
    args = p.parse_args()
    corpus = json.loads(args.corpus.read_text())
    reference = []
    if not args.golden_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
        for i, text in enumerate(H3):
            assert tokenizer.encode(text, add_special_tokens=False) == [151669+i], text
    for item in corpus:
        ids = item['ids'] if args.golden_only else tokenizer.encode(item['text'], add_special_tokens=False)
        assert ids == item['ids'], f"official metadata/golden disagreement: {item['name']}"
        reference.append(dict(name=item['name'], text=item['text'], ids=ids))
    def native(binary):
        return json.loads(subprocess.check_output([str(binary.resolve()),
            str(args.tokenizer/'tokenizer.json'), str(args.corpus)]))
    if args.native:
        current = native(args.native)
        assert len(current) == len(reference)
        for item, expected in zip(current, reference):
            assert item['name'] == expected['name']
            assert item['ids'] == expected['ids'], (item, expected)
            assert item['decoded'] == unicodedata.normalize('NFC', expected['text']), item
        if args.baseline:
            baseline = native(args.baseline)
            unchanged = 0
            for old, new in zip(baseline, current):
                if not any(marker in new['text'] for marker in H3):
                    assert old == new, (old, new)
                    unchanged += 1
            print(f'ok: {unchanged} ordinary/Qwen prompts unchanged from baseline')
        print(f'ok: {len(reference)} exact tokenizer parity and decode cases: {args.tokenizer}')
    else:
        print(json.dumps(reference, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
