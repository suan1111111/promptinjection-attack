"""Merge feature caches that share the same config fingerprint.

Creates a new directory whose features/ folder hard-links the source .npz files
and whose index.json concatenates the completed records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"D:\CodexProjects\PrivaSee")
SOURCES = [ROOT / "data/outputs/rcs_features_v2", ROOT / "data/outputs/rcs_features_benign_all"]
OUT = ROOT / "data/outputs/rcs_features_combined"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    configs = [json.loads((src / "config.json").read_text(encoding="utf-8")) for src in SOURCES]
    fingerprints = {c["fingerprint"] for c in configs}
    if len(fingerprints) != 1:
        raise SystemExit(f"fingerprint mismatch: {fingerprints}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(configs[0], ensure_ascii=False, indent=1), encoding="utf-8")

    records, seen = [], set()
    for src in SOURCES:
        index = json.loads((src / "index.json").read_text(encoding="utf-8"))
        for record in index["records"]:
            if record["status"] != "completed":
                continue
            if record["id"] in seen:
                raise SystemExit(f"duplicate id {record['id']}")
            seen.add(record["id"])
            source_file = src / record["file"]
            destination = out / "features" / source_file.name
            if not destination.exists():
                os.link(source_file, destination)
            record = dict(record)
            record["file"] = f"features/{destination.name}"
            records.append(record)
    payload = {
        "schema": 1,
        "config_fingerprint": configs[0]["fingerprint"],
        "manifest": "merged:" + ",".join(str(s) for s in SOURCES),
        "records": records,
    }
    (out / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("records", len(records), "features", len(list((out / "features").glob("*.npz"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
