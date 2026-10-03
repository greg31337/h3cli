# H3 token-fix validation

Validation uses local MiniMax-H3 FL2VA and Ref2VA assets on the Apple M4 Max
(128 GiB, macOS 26.6.2), against baseline `e4474ddedbd6c5741005b2ffb4b9371ffb11af41`.
The production change is confined to the shared tokenizer. Generated artifacts
and detailed logs are under `outputs/tokenfix-validation/`.

## Token correctness and compatibility

- All seven strings encode individually to exactly IDs 151669–151675 and decode
  from independently supplied golden IDs, without requiring model files.
- 209 mandatory conformance/collision checks pass, including identical existing
  mappings, string/ID conflicts in base and added vocabularies, direct adjacency,
  longer overlapping specials, sparse bounds, and the highest H3 ID.
- 230 checks exercise the actual FL2VA and Ref2VA presentation builders with a
  capture-only Qwen decoder: image/video numeric markers, ordered labels,
  timestamps, spans, modality tags, and reference embedding pointers are preserved.
- The original 60 Qwen-tokenizer checks pass for each installed model variant.
- All 35 corpus cases match the released local Hugging Face tokenizer exactly
  in both variants (Transformers 5.17.0). All 15 marker-free cases match the
  pre-fix native tokenizer exactly. No parity mismatches remain in the corpus.
- Opt-in diagnostics preserve encoded output and decoding; default stderr is
  empty. AddressSanitizer/UndefinedBehaviorSanitizer pass the conformance and
  multimodal tests. Compiling the new mandatory gate against the old tokenizer
  fails as expected (`count == n`).

The two installed tokenizer directories have identical JSON/config hashes,
recorded in `metadata.json`. `official-corpus.json`, `before-corpus.json`, and
`native-corpus.json` retain complete independently checked sequences.

The dialogue-generation prompt is:

```text
A woman looks at the camera and says:
<d>[English] Where are you going?</d>
```

Before (ordinary BPE fragments):

```text
[32, 5220, 5868, 518, 279, 6249, 323, 2727, 510, 90707, 30768,
 22574, 60, 10967, 525, 498, 2087, 26055, 67, 29]
```

Corrected (official model IDs):

```text
[32, 5220, 5868, 518, 279, 6249, 323, 2727, 510, 151669, 58,
 22574, 60, 10967, 525, 498, 2087, 30, 151670]
```

The full Qwen presentation IDs, including numeric vision tokens, are captured
separately for every generation in `generation/results.json` and `.ids` files.

## Generation protocol

`tests/tokenizer_generation.py` builds isolated before/after source copies and
runs eight full renders: control before/after, FL2VA dialogue before/after,
corrected dialogue with/without cutoff, and Ref2VA dialogue before/after.
Each uses seed 72, 124 frames, 256×256, 20 steps, all 50 DiT blocks, default CPU
Euler, and default native spatial RoPE. References and prompts are identical
within each pair. The FL2VA anchor is `inputs/face1.jpg`. Ref2VA combines that
face image with a silent 72-frame video made from `inputs/body1.jpg`.

All text, Qwen vision embeddings, and VAE reference conditions are freshly
encoded. No old conditioning is replayed into any corrected render. Instrumented
copies capture inputs, final audio/video latents, and hashes; instrumentation
does not enter production source. The complete commands and binary hashes are
in `generation/results.json`.

The marker-free control, `A woman looks at the camera and slowly walks across
the room.`, is **byte-identical before and after** at every captured stage:
token IDs, text embeddings, final audio/video latents, and encoded MP4. It
produces a coherent woman moving toward/across the camera. Baseline/corrected
render times were 207.8/206.2 seconds. This is a verified no-regression result.

### FL2VA dialogue (T022)

Before/after renders completed in 217.0/215.8 seconds. The actual multimodal
sequence changes from 92 to 91 rows, with the corrected prompt ending in the
19 official IDs shown above. Qwen merged/deepstack visual embeddings, image
spans, and the VAE image condition are byte-identical. Text conditioning and
final audio/video outputs change, as expected.

| Observation | Before | Corrected |
| --- | --- | --- |
| Local ASR | Extra opening speech, then “Where are you going?”, then extra trailing text | Exactly “Where are you going?” |
| Sound activity, peak-relative threshold | 0.02–5.16 s, multiple bursts | 2.92–3.62 s, one main utterance |
| Decoded audio RMS / peak | 0.0559 / 0.5962 | 0.0388 / 0.5336 |
| Visual review | Coherent face, frequent mouth movement | Coherent face, speech-related mouth movement around 3 s, closed mouth afterward |

The corrected face's inner-lip aspect ratio varies strongly during 3–4 s
(range 0.285), then settles during 4–5 s (range 0.011). This agrees coarsely
with the measured audio activity ending near 3.62 s. It does **not** establish
phoneme-perfect lip synchronization. Both videos retain the source face and
stable outdoor scene; no severe rendering corruption is visible in sampled
frames. The baseline ASR includes an endpoint beyond the actual 5.175-s decoded
audio; its exact segment times are unreliable and are flagged in the JSON.

This one controlled pair supports better adherence to the requested dialogue.
It is not a multi-seed quality benchmark. Review `generation/dialogue-before.mp4`
and `generation/dialogue-after.mp4`, their `.contact.png` frames and `.lips.json`
measurements, plus `audio-observations.json`.

### Cutoff (T023)

The matched corrected prompt is `A woman looks at the camera and says:` followed
by `<d>[English] Wait, don't—</d>`, with/without `<|cutoff|>`. The actual Qwen
sequence differs by exactly one appended ID, **151671**. All captured reference
embeddings and the VAE image condition match exactly; text and final AV outputs
change. Both renders complete (216.8/216.7 s) with coherent faces and no extra
spoken continuation in the local ASR result.

| Observation | With cutoff | Without cutoff |
| --- | --- | --- |
| Local ASR | “Wait, don't…” | “Wait, don't…” |
| Sound activity, peak-relative threshold | 1.88–3.26 s | 1.86–3.28 s |
| Decoded audio RMS / peak | 0.0401 / 0.4589 | 0.0413 / 0.4709 |

The cutoff render is consistent with interrupted speech. **An incremental
interruption effect from the token is not demonstrated:** the same already
unfinished phrase behaves similarly without it. The model receives the right
token, but this single pair does not prove stronger cutoff control. Raw ASR
segment endpoints differ more than actual waveform activity and should not be
used to claim an earlier cutoff. Review `generation/cutoff-after.mp4` and
`generation/cutoff-unmarked-after.mp4`.

### Ref2VA dialogue (T024)

Before/after renders completed in 327.0/331.3 seconds. Both use the released-v1
reference-video pipeline: 72 normalized frames, 56 VAE frames, 17 latent temporal
rows, and reference posterior seed 42. The main sampler seed remains 72.
The full Qwen presentation changes from 313 to 312 tokens and ends in the same
19 corrected prompt IDs as FL2VA. Its 293-token reference prefix and associated
three-axis positions are identical. All 16 Qwen visual arrays (image + three
video blocks, each with merged output and three deepstack outputs), VAE reference
condition, and spans match byte-for-byte.

| Observation | Before | Corrected |
| --- | --- | --- |
| Local ASR | “Where are you going?” | “Where are you going?” |
| Sound activity, peak-relative threshold | 0.24–5.14 s, including late activity | 1.18–2.58 s, one main interval |
| Decoded audio RMS / peak | 0.1027 / 1.2488 | 0.0938 / 1.1441 |
| Visual review | Coherent speaker; mouth opens again near the end | Coherent speaker; mouth settles after the utterance |

Both follow the body-video reference subject and room. The corrected view is
closer to the face; neither sampled sequence has the severe ghosting or broken
rendering seen in the earlier unrelated reference-video investigation. The
corrected clip's mouth motion roughly coincides with its main audio interval,
then settles during 3–5 s. Apple Vision found lip landmarks in 49/62 corrected
frames versus 62/62 baseline frames, limiting detailed automatic comparison.
AAC-decoded/resampled peaks exceed normalized full scale in both clips; this
change does not establish an audio-headroom improvement.

This confirms that corrected dialogue conditioning works with both image and
video references and leaves reference embedding construction intact. Both
transcripts already match, so there is no demonstrated word-accuracy gain in
this Ref2VA pair. Cleaner timing in one pair is not a broad lip-sync or video
quality guarantee.

## Acceptance and limits

All T001–T028 are addressed; T027's optional generalized metadata loader was
evaluated and deferred as explained in [the tokenizer guide](tokenizer.md).
`make -j8 test` passed, including the mandatory conformance gate and existing
host/Metal suites. Ten optional fixture groups were skipped because their
comparison fixtures are absent; those skips are listed in
`outputs/tokenfix-validation/full-suite.log`. Tokenizer correctness, both
released vocabularies, sanitizers, and all eight full model renders were tested.
`tests/tokenizer_generation_check.py` passes all structural and numerical
assertions, including exact control output and unchanged reference conditioning.

The tokenization defect is fixed, with exact official parity and no marker-free
regression in this corpus/control. The FL2VA dialogue pair supports improved
adherence to the requested words. An additional cutoff effect remains
inconclusive, and phoneme-level lip synchronization is not established. These
results do not fix or validate continuation boundaries or Ref2VA reference
takeover; those remain separate from tokenization.

## Review artifacts

| Pair | Before / without marker | Corrected / with marker |
| --- | --- | --- |
| Control | [Before](../outputs/tokenfix-validation/generation/control-before.mp4) | [Corrected](../outputs/tokenfix-validation/generation/control-after.mp4) |
| FL2VA dialogue | [Before](../outputs/tokenfix-validation/generation/dialogue-before.mp4) | [Corrected](../outputs/tokenfix-validation/generation/dialogue-after.mp4) |
| Cutoff | [Without cutoff](../outputs/tokenfix-validation/generation/cutoff-unmarked-after.mp4) | [With cutoff](../outputs/tokenfix-validation/generation/cutoff-after.mp4) |
| Ref2VA dialogue | [Before](../outputs/tokenfix-validation/generation/ref-dialogue-before.mp4) | [Corrected](../outputs/tokenfix-validation/generation/ref-dialogue-after.mp4) |

Each clip has a `.contact.png` sheet at 0, 1, 2, 3, 4, and 5 seconds. The JSON
observations preserve the ASR output, waveform measurements, and lip landmarks.
Artifacts are local, ignored test outputs; the test sources and this report are
versioned with the implementation.

## Reproduction

```sh
make -j8 test-tokenizer
make test-tokenizer-sanitize
make -j8 test
# In the installed local reference environment:
outputs/refvideo-encoder-validation/venv/bin/python tests/tokenizer_reference.py \
  models/MiniMax-H3/FL2VA/tokenizer --native ./bin/tokenizer_dump
outputs/refvideo-encoder-validation/venv/bin/python tests/tokenizer_reference.py \
  models/MiniMax-H3/Ref2VA/tokenizer --native ./bin/tokenizer_dump
python3 tests/tokenizer_generation.py --baseline e4474dd
python3 tests/tokenizer_generation_check.py
python3 tests/tokenizer_video_eval.py
outputs/refvideo-encoder-validation/venv/bin/python tests/tokenizer_audio_eval.py
```

Speech recognition uses a locally downloaded public `openai/whisper-base.en`
model (revision `911407f4214e0e1d82085af863093ec0b66f9cd6`), with no audio upload.
To install that optional test asset in a Python environment with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download('openai/whisper-base.en',
    revision='911407f4214e0e1d82085af863093ec0b66f9cd6',
    local_dir='outputs/tokenfix-validation/whisper-base.en',
    allow_patterns=['*.json', 'model.safetensors', 'vocab.*', 'merges.txt'])
```

Its transcript/timestamps are automated observations,
not a listening test. Apple Vision inner-lip aspect ratios at 12 fps provide
coarse mouth-motion observations, not phoneme-level synchronization scores.
