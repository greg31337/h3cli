## H3 Special-Token Tokenization Fix

### Problem

The current H3 tokenizer implementation loads special tokens from `tokenizer.json`, specifically from its `added_tokens` collection. MiniMax-H3 defines an additional set of model-specific special tokens in `tokenizer_config.json` that are not represented in the tokenizer data currently consumed by `h3cli`.

The affected tokens are:

```text
<d>                -> 151669
</d>               -> 151670
<|cutoff|>         -> 151671
<|lyrics_start|>   -> 151672
<|lyrics_end|>     -> 151673
<|caption_start|>  -> 151674
<|caption_end|>    -> 151675
```

Without explicit registration, `h3cli` processes these strings through ordinary BPE tokenization rather than emitting their dedicated model token IDs.

The primary known impact is incorrect text conditioning for dialogue and other H3-specific semantic markers. Ordinary prompts that do not contain these tokens are unaffected.

The change must preserve existing tokenization behavior for all existing vocabulary and special tokens while making the seven H3-specific tokens encode and decode exactly as expected by the released MiniMax-H3 model.

### Design Goals

The implementation should:

1. Produce the exact model token IDs defined by MiniMax-H3.
2. Preserve existing tokenization output for prompts not containing the new tokens.
3. Treat each H3 special token as an indivisible token regardless of surrounding text.
4. Support correct decoding of the added IDs.
5. Work identically for FL2VA and Ref2VA.
6. Avoid introducing a general dependency on the Hugging Face tokenizer runtime.
7. Verify behavior against the official MiniMax-H3 tokenizer.
8. Fail explicitly if future model/tokenizer data conflicts with the expected H3 token IDs.

### Recommended Implementation

The preferred implementation is to extend the existing `h3_tokenizer` initialization with the seven H3-defined tokens after loading `tokenizer.json`.

Do not dynamically assign new IDs. The IDs are part of the trained model vocabulary and must remain exactly:

```text
151669 .. 151675
```

The registration operation should update all tokenizer structures used for:

* special-token recognition during encoding;
* token-to-ID lookup;
* ID-to-token decoding;
* maximum token-ID / vocabulary-bound calculations;
* any structures used by tokenizer debugging or tests.

The special-token scanner should recognize the complete string before ordinary BPE processing.

For example:

```text
"She says <d>Hello</d>."
```

must conceptually tokenize as:

```text
ordinary tokens for "She says "
151669
ordinary tokens for "Hello"
151670
ordinary tokens for "."
```

rather than allowing `<d>` or `</d>` to be split by BPE.

### Source of Token Definitions

For the current MiniMax-H3 model generation, the seven tokens may be represented in a small static table in the H3 tokenizer implementation:

```c
struct h3_extra_special_token {
    const char *text;
    uint32_t id;
};

static const struct h3_extra_special_token h3_special_tokens[] = {
    {"<d>",               151669},
    {"</d>",              151670},
    {"<|cutoff|>",        151671},
    {"<|lyrics_start|>",  151672},
    {"<|lyrics_end|>",    151673},
    {"<|caption_start|>", 151674},
    {"<|caption_end|>",   151675},
};
```

This is preferable to assigning IDs based on the end of the vocabulary because these IDs correspond to trained embedding rows.

A later generalization may parse `tokenizer_config.json`, but that is not required for the initial correction.

### Optional Generalized Implementation

If avoiding hard-coded model metadata is preferred, `h3_tokenizer_load()` may additionally accept the path to `tokenizer_config.json` and load `additional_special_tokens`.

However, `tokenizer_config.json` alone may specify token strings without being sufficient to derive the intended model IDs safely.

Therefore, even in the generalized implementation, H3 should validate that:

```text
<d>                == 151669
</d>               == 151670
<|cutoff|>         == 151671
<|lyrics_start|>   == 151672
<|lyrics_end|>     == 151673
<|caption_start|>  == 151674
<|caption_end|>    == 151675
```

for the supported H3 model.

If loaded model metadata disagrees with these values, initialization should fail rather than silently assigning different IDs.

### Token Matching

Special-token matching must happen before normal BPE processing.

The matching implementation should correctly handle:

```text
<d>Hello</d>
foo<d>bar</d>baz
<d>
</d>
<|cutoff|>
```

without requiring whitespace around the token.

Existing behavior for overlapping or prefix-like special tokens must not regress.

The implementation should preferably use the tokenizer's existing special-token recognition mechanism rather than creating a second independent preprocessing path.

### Decode Support

The existing ID-to-token representation must be extended through at least ID `151675`.

Calling the tokenizer decoder with:

```text
151669
```

must produce:

```text
<d>
```

and similarly for the other six tokens.

The implementation must verify that internal arrays indexed by token ID are sized based on the maximum effective token ID after H3-specific tokens have been registered, rather than only the maximum ID originally present in `tokenizer.json`.

### FL2VA and Ref2VA

The correction applies to both FL2VA and Ref2VA because both pipelines use H3 text conditioning.

No changes are required to Ref2VA visual-token construction. Vision markers such as `vision_start`, `vision_end`, `image_pad`, and `video_pad` are inserted independently using their numeric token IDs and are therefore outside the scope of this change.

The change should occur at the shared tokenizer layer so both modes receive identical behavior.

### Compatibility

For any prompt that does not contain one of the seven new strings:

```text
encode_before(prompt) == encode_after(prompt)
```

must hold exactly.

This should be tested over a representative corpus of existing H3 prompts.

The fix must therefore not:

* renumber any existing token;
* change BPE merge behavior;
* alter normalization;
* change ordinary Unicode handling;
* modify existing Qwen special-token handling;
* change multimodal token insertion;
* affect random seeds or sampling directly.

Videos generated from prompts without these markers should receive byte-for-byte identical token-ID sequences before and after the change.

### Reference Verification

A small Python reference utility should use the official MiniMax-H3 Hugging Face tokenizer and emit token IDs for test strings.

The C tokenizer output should then be compared against the Python output.

Required parity cases include:

```text
<d>
</d>
<|cutoff|>
<|lyrics_start|>
<|lyrics_end|>
<|caption_start|>
<|caption_end|>

<d>Hello</d>

A woman says: <d>[English] Hello there.</d>

A man says: <d>[English] Stop.</d><|cutoff|>

prefix<d>text</d>suffix
```

The test should compare exact integer token sequences, not decoded text alone.

### Regression Testing

Add tests covering both the seven new tokens and existing tokenizer behavior.

At minimum verify:

```text
encode("<d>")                == [151669]
encode("</d>")               == [151670]
encode("<|cutoff|>")         == [151671]
encode("<|lyrics_start|>")   == [151672]
encode("<|lyrics_end|>")     == [151673]
encode("<|caption_start|>")  == [151674]
encode("<|caption_end|>")    == [151675]
```

Also verify round-trip decoding:

```text
decode([151669]) == "<d>"
...
decode([151675]) == "<|caption_end|>"
```

Existing tests such as `<|im_start|>` must continue to pass.

Add mixed-token tests ensuring special tokens embedded directly next to ordinary text are recognized correctly.

### Behavioral Validation

In addition to tokenizer unit tests, perform generation tests using identical:

* model;
* seed;
* resolution;
* step count;
* scheduler;
* prompt;
* Ref2VA inputs where applicable.

Use at least three prompt classes.

#### Control prompt

No new special tokens:

```text
A woman looks at the camera and slowly walks across the room.
```

Expected result:

```text
token IDs before fix == token IDs after fix
```

Generation should consequently remain unchanged apart from unrelated nondeterminism.

#### Dialogue prompt

```text
A woman looks at the camera and says:
<d>[English] Where are you going?</d>
```

Expected result:

* token sequence differs from the old implementation;
* `<d>` and `</d>` become IDs 151669 and 151670;
* audio/dialogue generation should use the model's intended conditioning.

#### Dialogue with cutoff

```text
A man says:
<d>[English] Wait, don't—</d><|cutoff|>
```

Verify the dedicated cutoff token is emitted and evaluate resulting speech behavior.

### Logging / Diagnostics

When verbose tokenizer diagnostics are enabled, optionally expose recognized special tokens in a form such as:

```text
special token: "<d>" -> 151669
special token: "</d>" -> 151670
```

Do not emit this at normal verbosity.

A tokenizer debug option may additionally print the final token sequence for a prompt to simplify future model-parity investigations.

### Out of Scope

This change does not attempt to:

* change H3 prompt syntax;
* automatically insert `<d>` around dialogue;
* reinterpret `<cutoff>` as `<|cutoff|>`;
* change Ref2VA visual conditioning;
* address video continuation behavior;
* address temporal chunk boundaries;
* solve reference-image takeover near the end of generated videos;
* reproduce the entire Hugging Face tokenizer configuration system.

Those should remain independent features or investigations.
