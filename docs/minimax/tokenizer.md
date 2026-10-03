# MiniMax-H3 prompt tokens

The shared native tokenizer recognizes these exact model tokens in FL2VA and
Ref2VA, including when directly adjacent to ordinary text or other markers:

| String | Trained token ID |
| --- | ---: |
| `<d>` | 151669 |
| `</d>` | 151670 |
| `<\|cutoff\|>` | 151671 |
| `<\|lyrics_start\|>` | 151672 |
| `<\|lyrics_end\|>` | 151673 |
| `<\|caption_start\|>` | 151674 |
| `<\|caption_end\|>` | 151675 |

For example, `<d>[English] Hello there.</d>` encodes to
`[151669, 58, 22574, 60, 21927, 1052, 13, 151670]`.
The string `<cutoff>` is ordinary prompt text. It is **not** an alias for the
model token `<|cutoff|>`. Dialogue markers are never inserted automatically.

The released `tokenizer.json` contains added tokens only through ID 151668.
`tokenizer_config.json` lists the seven H3 strings in `additional_special_tokens`;
the released Hugging Face tokenizer assigns them IDs 151669–151675. Native
initialization registers a fixed canonical table in the existing special-token
scanner before computing vocabulary bounds. These are trained embedding rows,
so allocating IDs from the end of an arbitrary vocabulary would be incorrect.
The decoder also recognizes every marker. Both base-vocabulary and added-token
collisions are initialization errors; an identical string/ID mapping is accepted.

BPE merges, normalization, ordinary Unicode handling, Qwen tokens, and numeric
vision insertion are unchanged. The longest special token at the earliest
matching location still takes precedence. No Hugging Face runtime dependency
is added to native inference.

## Diagnostics

Normal output is unchanged. To print recognized special-token strings and IDs
to stderr, set `H3_DEBUG_TOKENIZER=1`:

```sh
H3_DEBUG_TOKENIZER=1 ./bin/h3cli -d models/MiniMax-H3 \
  -p 'A woman says: <d>[English] Hello there.</d>' \
  --width 256 --height 256 --frames 124 --steps 20 -o outputs/dialogue.mp4
```

The variable uses the existing diagnostic environment namespace, so enabling it
does not change the numerical environment recorded by sampler checkpoints.
`bin/tokenizer_dump` prints complete IDs and decoded strings for a JSON corpus.

## Tests and reference verification

```sh
make -j8 test-tokenizer
make test-tokenizer-sanitize
make -j8 test
```

The normal test suite **always** runs the seven-token conformance, collision,
and multimodal presentation tests against a synthetic byte vocabulary. Model
weights and Python packages are unnecessary for this mandatory gate. If local
FL2VA/Ref2VA tokenizer files exist, it also runs the original Qwen checks and
35 exact golden cases in `tests/tokenizer_corpus.json` against each variant.

For an independent comparison, use a Python environment with `transformers`
(the validation environment used 5.17.0). All model files are loaded locally:

```sh
python tests/tokenizer_reference.py models/MiniMax-H3/FL2VA/tokenizer
python tests/tokenizer_reference.py models/MiniMax-H3/FL2VA/tokenizer --native ./bin/tokenizer_dump
python tests/tokenizer_reference.py models/MiniMax-H3/Ref2VA/tokenizer --native ./bin/tokenizer_dump
```

The reference utility rejects any official H3 mapping or golden-ID disagreement.
`--baseline PATH_TO_OLD_DUMP_BINARY` additionally checks exact before/after
compatibility for the 15 corpus prompts without H3 markers. `--golden-only`
uses the committed reference IDs without importing `transformers`.

Full generation validation, including all commands, fresh conditioning hashes,
actual Qwen presentation IDs, reference embeddings and final latents:

```sh
python3 tests/tokenizer_generation.py --baseline e4474dd
# Resume completed cases without rebuilding the isolated executables:
python3 tests/tokenizer_generation.py --resume
```

This runs eight 124-frame, 256-square, 20-step, seed-72 renders using all 50 DiT
layers and the default CPU Euler scheduler on the local machine. It compares
plain control and FL2VA dialogue before/after, compares corrected cutoff against
the same corrected dialogue without cutoff, and compares Ref2VA dialogue with
both a face image and a silent body-reference video. It never replays old text
conditioning into a corrected dialogue run. Instrumentation exists only in
isolated test copies. See [the validation report](tokenfix-validation.md) for
results and the limits of the behavioral evidence.

## Scope and possible generalization

This fix is independent of continuation temporal conditioning, chunk boundaries,
and Ref2VA reference-image takeover or previously broken video renders. Prompts
without the seven markers retain exactly the same token IDs. Better dialogue
conditioning does not establish a fix for those separate visual problems.

T027 evaluated reading `tokenizer_config.json` at runtime. It is deferred: its
`additional_special_tokens` is an ordered list of strings, not a reliable
explicit ID map, and the seven H3 entries are absent from its released
`added_tokens_decoder`. The small validated canonical table is sufficient for
the supported H3 model. A future generalized loader must validate explicit H3
IDs against this table and fail initialization on disagreement; it must never
silently allocate different embedding rows. The current native loader consumes
`tokenizer.json`, not a second configuration file.
