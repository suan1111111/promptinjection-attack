---
license: mit
language:
- en
pretty_name: RI Multimodal Prompt Injection Dataset
tags:
- image
- audio
- multimodal
- prompt-injection
- safety
configs:
- config_name: audio
  data_files:
  - split: calibration
    path: data/audio/calibration-*.parquet
  - split: dev
    path: data/audio/dev-*.parquet
  - split: external
    path: data/audio/external-*.parquet
  - split: hard_benign
    path: data/audio/hard_benign-*.parquet
  - split: policy
    path: data/audio/policy-*.parquet
  - split: test
    path: data/audio/test-*.parquet
  - split: train
    path: data/audio/train-*.parquet
- config_name: image
  data_files:
  - split: calibration
    path: data/image/calibration-*.parquet
  - split: dev
    path: data/image/dev-*.parquet
  - split: external
    path: data/image/external-*.parquet
  - split: hard_benign
    path: data/image/hard_benign-*.parquet
  - split: policy
    path: data/image/policy-*.parquet
  - split: test
    path: data/image/test-*.parquet
  - split: train
    path: data/image/train-*.parquet
- config_name: native_image_audio
  data_files:
  - split: calibration
    path: data/native_image_audio/calibration-*.parquet
  - split: dev
    path: data/native_image_audio/dev-*.parquet
  - split: policy
    path: data/native_image_audio/policy-*.parquet
  - split: test
    path: data/native_image_audio/test-*.parquet
  - split: train
    path: data/native_image_audio/train-*.parquet
---
# RI Multimodal Prompt Injection Dataset

This dataset contains image, audio, and paired image/audio samples used to test and evaluate [UMIS — Unified Multimodal Injection Sentinel](https://github.com/hanshi-798/umis). It supports testing multimodal prompt-injection detection and the handling of benign inputs.

## Companion tool: UMIS

[UMIS — Unified Multimodal Injection Sentinel](https://github.com/hanshi-798/umis) is a research tool for inspecting prompt injection carried by images and audio before that media reaches a multimodal assistant. It combines file/signal probes, OCR/ASR extraction, and optional semantic features to produce `BLOCK`, `QUARANTINE+CONFIRM`, or `ALLOW+AUDIT` decisions, with localized evidence and extracted text. The receiving application is responsible for enforcing those decisions.

The repository includes the Python detector and CLI, image/audio attack generators, benchmark construction, and calibration/evaluation tools. This Hugging Face repository supplies research data; the GitHub repository supplies the tool code. See the [English usage instructions](https://github.com/hanshi-798/umis#use) or [中文说明](https://github.com/hanshi-798/umis/blob/main/README.zh-CN.md) for installation, optional dependencies, and usage.

For downloaders: media is embedded in the `image` and `audio` Parquet columns. Export the relevant media bytes to local files before passing file paths to the UMIS CLI. The benchmark commands in the tool repository use their own manifest format; this dataset's JSONL/Parquet release is not asserted to be a drop-in replacement. Optional calibrated inference requires matching model artifacts, which are not included in this dataset. 

## Configurations and splits

| Configuration | Split | Samples |
|---|---|---:|
| audio | calibration | 22 |
| audio | dev | 36 |
| audio | external | 120 |
| audio | hard_benign | 10 |
| audio | policy | 16 |
| audio | test | 82 |
| audio | train | 210 |
| image | calibration | 50 |
| image | dev | 100 |
| image | external | 250 |
| image | hard_benign | 100 |
| image | policy | 50 |
| image | test | 200 |
| image | train | 600 |
| native_image_audio | calibration | 8 |
| native_image_audio | dev | 24 |
| native_image_audio | policy | 14 |
| native_image_audio | test | 38 |
| native_image_audio | train | 150 |

Original `train`, `dev`, `test`, `policy`, and `calibration` partitions are preserved. Records with a null original partition are placed in separate `external` and `hard_benign` splits according to their source set; this does not assign them to training or test. The source `partition` remains null for those rows. No resampling or new benchmark split was performed.

```python
from datasets import load_dataset

ds = load_dataset("hanIshi/ri-multimodal-injection", "image")
audio = load_dataset("hanIshi/ri-multimodal-injection", "audio")
joint = load_dataset("hanIshi/ri-multimodal-injection", "native_image_audio")
sample = joint["train"][0]
print(sample["sample_id"], sample["is_attack"])
```

Each Parquet record has Hugging Face `Image` and `Audio` features with embedded, byte-identical source media. The unused modality is null. Modern audio decoding may require the decoder dependencies supported by the installed `datasets` version. Streaming is available through `streaming=True`.

## Fields and preservation

- `sample_id`, `pair_id`, `component_id`, `source_group_id`: sample, pairing, component, and source identifiers.
- `is_attack`, `attack_family`, `family`, `benign_category`: original labels; missing `family` is preserved.
- `attack_goal`, `media_text`, `image_part`, `audio_part`, `joint_text`: generated text and injection content.
- `trusted_task_id`, `task_topic`, `gold_task_answer`: task identifiers and expected-answer descriptions. The supplied manifest does not contain a separate full trusted-task prompt.
- `set`, `partition`, `split`: source set, original partition, and release split.
- `media_path`, `image_path`, `audio_path`: paths relative to this repository, replacing the old Linux absolute paths.
- `media_sha256`: original hash of `media_path`; for joint rows this hashes the primary image, not both modalities. `media_checksums.json` provides hashes for every released file, including joint audio.
- `license`, source assets, speaker/clip identifiers, generator seed and revision: preserved source metadata.

Nested fields are stored as JSON strings in the Parquet columns with a `_json` suffix, to avoid incompatible optional nested structures across modalities. Parse them with `json.loads`. The lossless structured versions remain in `manifest.jsonl`. `schema.json` lists Parquet feature definitions. Render-recipe font paths are historical generation metadata, not required local dependencies.

This Hub upload embeds all referenced media in Parquet. The standalone media/ directory is retained locally and is not uploaded; media_path, image_path and audio_path describe the local release layout and are not downloadable raw-file links in this Hub repository. The manifest and checksum files are included for provenance. The archive contains additional media and experiment artifacts, but this release includes only assets referenced by the supplied 2,080-row manifest. Model weights, OCR/ASR caches, result files, invalidated checkpoints and migration scripts are excluded.

## License

Released under the [MIT License](LICENSE). Third-party source media retain their original licenses, recorded in the per-sample `license` field.
