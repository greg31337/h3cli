# Model file dependencies

The [catalog](../../src/models/catalog.json) is the authoritative file-level
manifest: exact source URL, immutable revision, byte size and full SHA-256 for
119 files. The resolver is [resolve.c](../../src/models/resolve.c).
No repository Python files or alternative Diffusers layouts are downloaded.

| Effective operation | Required main files | Auxiliary files |
| --- | --- | --- |
| Text, first/last anchors | 57 FL2VA files | None |
| Any image, audio or video reference | 57 Ref2VA files | None |
| Still generation | Selected mode's 57 files | Image VAE |
| Still latent decode | None | Image VAE |
| AV decode, full or preview | Eight identity files listed below, in saved mode | Preview VAE only with effective preview |
| Hard continuation / bridge | Selected generation mode's 57 files | Effective decoder |
| Sampler resume | Saved mode's 57 files | Current permitted preview override |
| Fresh upscale | Saved source mode's 57 files | BF16 learned upscaler |
| Upscale sampler resume | Saved mode's 57 files | None; transfer is already in the checkpoint |
| State-only or stopped generation | Same main closure (startup inventory and identity) | Only selected, validated decoder; preview controls retain native admission rules |
| `--show` with full decoding | Same main closure | None |
| Help, info, state inspection, validate-only | Local lookup only | Never download |
| Prefetch `base` / `references` | Mode's 57 files plus LICENSE and modular index | None |
| Prefetch auxiliary group | None | Only that group's one file |

`preview` and `fast-preview` quality enable the tiny decoder; explicit
`--no-preview-vae` disables it regardless of quality flag order. Last explicit
preview negation wins. Precision, reuse, adaptive cache, SubBlock and residency
do not change the source-weight closure. LoRA inputs are local user resources.

The eight AV decode identity files are `transformer/config.json`,
`video_vae/config.json`, `video_vae/source/config.json`,
`video_vae/source/model.safetensors`, `audio_vae/config.json`,
`audio_vae/config.yaml`, `audio_vae/metadata.json` and
`audio_vae/model.safetensors`, beneath the saved FL2VA or Ref2VA mode.
The current compatibility signature includes both full VAEs even for preview
decode; retaining these dependencies preserves saved-state checks.

Generation startup inventories the selected text encoder, transformer and both
VAEs. Resume/upscale reuse saved conditioning but keep this inventory and the
existing recursive model identity. An identity-only AV decode installation is
not selected as a generation checkpoint: it has no tokenizer. This prevents
an earlier FL2VA decode from shadowing a later complete Ref2VA installation.

Each main mode is 144,051,030,183 / 144,051,030,171 bytes including notices.
All groups total **294,023,151,343 logical bytes**. Content deduplication reduces
an empty installation to **216,229,637,924 unique transfer bytes**. CoW saves
space when supported; capacity checks allow ordinary copy fallback on each
filesystem. Partial-file blocks already stored are deducted from the remaining
space requirement. Runtime quantization caches and outputs require extra space.

## File-level inventory

`base` and `references` entries belong to the respective full main closure.
Tokenizer/config files are needed for loading/identity; safetensors files are
needed for computation or broad native startup inventory. `base,references`
entries are prefetch notices/index, outside runtime identity scans.

| Destination relative to component root | Groups | Bytes |
| --- | --- | ---: |
| `FL2VA/audio_vae/config.json` | base | 1973 |
| `FL2VA/audio_vae/config.yaml` | base | 91 |
| `FL2VA/audio_vae/metadata.json` | base | 440 |
| `FL2VA/audio_vae/model.safetensors` | base | 605429308 |
| `FL2VA/model_index.json` | base | 719 |
| `FL2VA/processor/chat_template.json` | base | 5499 |
| `FL2VA/processor/merges.txt` | base | 1671839 |
| `FL2VA/processor/preprocessor_config.json` | base | 390 |
| `FL2VA/processor/tokenizer.json` | base | 7032403 |
| `FL2VA/processor/tokenizer_config.json` | base | 11003 |
| `FL2VA/processor/video_preprocessor_config.json` | base | 385 |
| `FL2VA/processor/vocab.json` | base | 2776833 |
| `FL2VA/text_encoder/chat_template.json` | base | 5499 |
| `FL2VA/text_encoder/config.json` | base | 1474 |
| `FL2VA/text_encoder/merges.txt` | base | 1671839 |
| `FL2VA/text_encoder/model-00001-of-00014.safetensors` | base | 4932328944 |
| `FL2VA/text_encoder/model-00002-of-00014.safetensors` | base | 4875990528 |
| `FL2VA/text_encoder/model-00003-of-00014.safetensors` | base | 4875990552 |
| `FL2VA/text_encoder/model-00004-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00005-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00006-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00007-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00008-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00009-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00010-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00011-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00012-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00013-of-00014.safetensors` | base | 4875990584 |
| `FL2VA/text_encoder/model-00014-of-00014.safetensors` | base | 3270697008 |
| `FL2VA/text_encoder/model.safetensors.index.json` | base | 97831 |
| `FL2VA/text_encoder/preprocessor_config.json` | base | 390 |
| `FL2VA/text_encoder/tokenizer.json` | base | 7032403 |
| `FL2VA/text_encoder/tokenizer_config.json` | base | 11003 |
| `FL2VA/text_encoder/video_preprocessor_config.json` | base | 385 |
| `FL2VA/text_encoder/vocab.json` | base | 2776833 |
| `FL2VA/tokenizer/merges.txt` | base | 1671839 |
| `FL2VA/tokenizer/tokenizer.json` | base | 7032403 |
| `FL2VA/tokenizer/tokenizer_config.json` | base | 11003 |
| `FL2VA/tokenizer/vocab.json` | base | 2776833 |
| `FL2VA/transformer/config.json` | base | 604 |
| `FL2VA/transformer/model-00001-of-00013.safetensors` | base | 5227812968 |
| `FL2VA/transformer/model-00002-of-00013.safetensors` | base | 5164578856 |
| `FL2VA/transformer/model-00003-of-00013.safetensors` | base | 5164578872 |
| `FL2VA/transformer/model-00004-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00005-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00006-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00007-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00008-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00009-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00010-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00011-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00012-of-00013.safetensors` | base | 5164578896 |
| `FL2VA/transformer/model-00013-of-00013.safetensors` | base | 4242305176 |
| `FL2VA/transformer/model.safetensors.index.json` | base | 38323 |
| `FL2VA/video_vae/config.json` | base | 1807 |
| `FL2VA/video_vae/source/config.json` | base | 1164 |
| `FL2VA/video_vae/source/model.safetensors` | base | 10415548320 |
| `LICENSE` | base,references | 17604 |
| `Ref2VA/audio_vae/config.json` | references | 1973 |
| `Ref2VA/audio_vae/config.yaml` | references | 91 |
| `Ref2VA/audio_vae/metadata.json` | references | 440 |
| `Ref2VA/audio_vae/model.safetensors` | references | 605429308 |
| `Ref2VA/model_index.json` | references | 707 |
| `Ref2VA/processor/chat_template.json` | references | 5499 |
| `Ref2VA/processor/merges.txt` | references | 1671839 |
| `Ref2VA/processor/preprocessor_config.json` | references | 390 |
| `Ref2VA/processor/tokenizer.json` | references | 7032403 |
| `Ref2VA/processor/tokenizer_config.json` | references | 11003 |
| `Ref2VA/processor/video_preprocessor_config.json` | references | 385 |
| `Ref2VA/processor/vocab.json` | references | 2776833 |
| `Ref2VA/text_encoder/chat_template.json` | references | 5499 |
| `Ref2VA/text_encoder/config.json` | references | 1474 |
| `Ref2VA/text_encoder/merges.txt` | references | 1671839 |
| `Ref2VA/text_encoder/model-00001-of-00014.safetensors` | references | 4932328944 |
| `Ref2VA/text_encoder/model-00002-of-00014.safetensors` | references | 4875990528 |
| `Ref2VA/text_encoder/model-00003-of-00014.safetensors` | references | 4875990552 |
| `Ref2VA/text_encoder/model-00004-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00005-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00006-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00007-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00008-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00009-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00010-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00011-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00012-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00013-of-00014.safetensors` | references | 4875990584 |
| `Ref2VA/text_encoder/model-00014-of-00014.safetensors` | references | 3270697008 |
| `Ref2VA/text_encoder/model.safetensors.index.json` | references | 97831 |
| `Ref2VA/text_encoder/preprocessor_config.json` | references | 390 |
| `Ref2VA/text_encoder/tokenizer.json` | references | 7032403 |
| `Ref2VA/text_encoder/tokenizer_config.json` | references | 11003 |
| `Ref2VA/text_encoder/video_preprocessor_config.json` | references | 385 |
| `Ref2VA/text_encoder/vocab.json` | references | 2776833 |
| `Ref2VA/tokenizer/merges.txt` | references | 1671839 |
| `Ref2VA/tokenizer/tokenizer.json` | references | 7032403 |
| `Ref2VA/tokenizer/tokenizer_config.json` | references | 11003 |
| `Ref2VA/tokenizer/vocab.json` | references | 2776833 |
| `Ref2VA/transformer/config.json` | references | 604 |
| `Ref2VA/transformer/model-00001-of-00013.safetensors` | references | 5227812968 |
| `Ref2VA/transformer/model-00002-of-00013.safetensors` | references | 5164578856 |
| `Ref2VA/transformer/model-00003-of-00013.safetensors` | references | 5164578872 |
| `Ref2VA/transformer/model-00004-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00005-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00006-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00007-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00008-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00009-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00010-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00011-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00012-of-00013.safetensors` | references | 5164578896 |
| `Ref2VA/transformer/model-00013-of-00013.safetensors` | references | 4242305176 |
| `Ref2VA/transformer/model.safetensors.index.json` | references | 38323 |
| `Ref2VA/video_vae/config.json` | references | 1807 |
| `Ref2VA/video_vae/source/config.json` | references | 1164 |
| `Ref2VA/video_vae/source/model.safetensors` | references | 10415548320 |
| `modular_model_index.json` | base,references | 2935 |
| `preview-vae/taeh3.safetensors` | preview | 22709752 |
| `image-vae/minimax_h3_t1_image_vae_step1597.safetensors` | image-vae | 5207808784 |
| `latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | upscale | 690592992 |
