# RCS source provenance

Authors: Peichun Hua and collaborators.

Repository: https://github.com/sarendis56/Jailbreak_Detection_RCS

Pinned revision: `3ea8e29f3542e7962a7eaed7996dc5654b4035fd`.

Original source URLs:

- https://github.com/sarendis56/Jailbreak_Detection_RCS/blob/3ea8e29f3542e7962a7eaed7996dc5654b4035fd/code/kcd.py
- https://github.com/sarendis56/Jailbreak_Detection_RCS/blob/3ea8e29f3542e7962a7eaed7996dc5654b4035fd/code/feature_extractor_qwen.py

The `.py.txt` files are source snapshots, not imported experiment entry points.
`backend/rcs_projection_core.py` copies `ProjectionConfig`, `LearnedProjection`,
and `compute_contrastive_loss` unchanged, with only standalone imports and a
Qwen configuration instance added outside those definitions. The pipeline tests
compare their ASTs with the pinned snapshot.

See `docs/rcs_feature_projection.md` for the training-loop and feature-extraction
adaptations. These snapshots and attribution do not imply that this project
reproduces the paper's detection results.
