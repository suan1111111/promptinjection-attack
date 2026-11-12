"""Build a class-balanced feature cache from the combined v2 + benign features.

Keeps all malicious training rows and samples benign rows down to the same count,
with roughly half from the paired text benign set and half from the clean COCO pool.
Validation/test rows are kept untouched.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(r"D:\CodexProjects\PrivaSee")
SOURCE = ROOT / "data/outputs/rcs_features_combined"
OUT = ROOT / "data/outputs/rcs_features_balanced"
CLEAN_SHARE = 0.5


def main() -> int:
    index = json.loads((SOURCE / "index.json").read_text(encoding="utf-8"))
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    train = [r for r in index["records"] if r["status"] == "completed" and r["row"]["split"] == "train"]
    others = [r for r in index["records"] if r["status"] == "completed" and r["row"]["split"] != "train"]
    malicious = [r for r in train if r["row"]["label"] == 1]
    benign_text = [r for r in train if r["row"]["label"] == 0 and r["row"]["dataset"] != "benign_coco"]
    benign_clean = [r for r in train if r["row"]["label"] == 0 and r["row"]["dataset"] == "benign_coco"]

    target = len(malicious)
    clean_target = min(len(benign_clean), int(round(target * CLEAN_SHARE)))
    text_target = target - clean_target
    if text_target > len(benign_text):
        text_target = len(benign_text)
        clean_target = min(len(benign_clean), target - text_target)

    rng = random.Random(42)
    sampled = (rng.sample(benign_text, text_target) + rng.sample(benign_clean, clean_target))
    selected = malicious + sampled

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    records = []
    for record in selected + others:
        source_file = SOURCE / record["file"]
        destination = OUT / "features" / source_file.name
        if not destination.exists():
            os.link(source_file, destination)
        copy = dict(record)
        copy["file"] = f"features/{destination.name}"
        records.append(copy)
    payload = {
        "schema": 1,
        "config_fingerprint": config["fingerprint"],
        "manifest": f"balanced:{SOURCE}",
        "records": records,
    }
    (OUT / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"malicious={len(malicious)} benign_text={len(sampled)-clean_target} benign_clean={clean_target} "
          f"total_train={len(selected)} others={len(others)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
