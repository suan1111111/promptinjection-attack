# Seven-form Qwen pilot

Run from the project root:

```powershell
.\.venv\Scripts\python.exe pixel_attack_trials/run_seven_previews.py --work-dir outputs/seven_attack_preview_20261026 --steps 300
```

The pilot pairs `clean_images/0.png` with `instruction_data/0/Attack/dataset.csv` in the local Soft-Prompts-Go-Hard checkout. All seven jobs preserve that pairing and optimize the complete `injection` answer. No target instruction is inserted in evaluation questions. The target literal is `<maliciousurl>`; no live URL is used.

The shared CSV reader accepts the existing split/question/target_answer schema and the upstream Question/normal/injection schema. Upstream rows 1–40 train, 41–50 validate (reserved), and 51 onwards test. Only training rows enter gradient updates. The pilot evaluates the first five test questions after reloading the saved 8-bit images. These are preliminary test observations, not an independent final evaluation after parameter selection.

All jobs use the local frozen Qwen2.5-VL-3B-Instruct NF4 model, 128 visual tokens, seed 7, one question per update, 300 updates, and a sign step of 1/255. Full-image and low-frequency epsilon is 16/255; border and sparse epsilon is 32/255. Border width is 8 resized pixels; sparse support is at most 256 RGB positions. Flower is a mathematical five-petal mask in a 56×56 box (not a natural flower photograph). Rectangular patches are 56×56, both random-initialized, without epsilon constraints. Fixed placement is top right; random training samples legal positions each update.

`final_images/` contains only the final composed PNG for each completed job. It does not contain standalone patches, clean images, logs, or metadata. Numbered names identify global, border, lowfreq, flower, sparse, fixed_patch, random_patch. Logs, checkpoints, evaluations, original input hashes and process state stay in the sibling job directories and manifest/status JSON files.

For random patches the exported image uses the fixed top-right preview position; this pilot evaluates that exported placement only. It does not measure generalization across positions. The varying perturbation budgets are method-specific pilot settings, not an equal-budget ranking. Completed image generation is not evidence of successful instruction injection; examine `evaluation.json` clean/adversarial answers and literal marker matches.

The serial runner skips verified completed states, resumes pixel checkpoints, and retries unsuccessful patch jobs in new timestamped directories. Failed jobs are recorded and do not prevent the remaining forms from being attempted. Reuse the exact same command to retry; select a new work directory when changing inputs/settings.
