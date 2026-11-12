"""Build a clean-photo benign pool from the COCO backgrounds used by RI.

For every text pair in rcs_train_v2 we look up its COCO source asset, download the
text-free original from COCO val2017, and emit a benign manifest row. These are
"clean background" negatives: same scenes as the text-injection pairs but without
any rendered text.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\CodexProjects\PrivaSee")
RI = ROOT / "data/datasets/ri_multimodal_injection"
V2 = ROOT / "data/datasets/rcs_train_v2"
OUT = ROOT / "data/datasets/benign_coco"
VAL_URL = "http://images.cocodataset.org/val2017/{asset}.jpg"
TRAIN_URL = "http://images.cocodataset.org/train2017/{asset}.jpg"


def fetch(asset: str, destination: Path) -> str | None:
    if destination.is_file() and destination.stat().st_size > 1024:
        return "cached"
    for url in (VAL_URL.format(asset=asset), TRAIN_URL.format(asset=asset)):
        for _ in range(1):
            try:
                data = urllib.request.urlopen(url, timeout=20).read()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                return url.split("/")[-2]
            except Exception:  # noqa: BLE001
                continue
    return None


def main() -> int:
    pairs = json.loads((V2 / "pairs.json").read_text(encoding="utf-8"))
    text_pairs = [p for p in pairs if p.get("domain") == "text"]
    assets = sorted({
        p["group"].split(":", 1)[1]
        for p in text_pairs
        if str(p.get("group", "")).startswith("coco:")
    })
    print(f"text pairs={len(text_pairs)} unique coco assets={len(assets)}")
    manifest = []
    failed = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    v2_rows = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    asset_splits: dict[str, set[str]] = {}
    for entry in v2_rows:
        group = str(entry.get("group", ""))
        if entry.get("domain") != "text" or not group.startswith("coco:"):
            continue
        asset_splits.setdefault(group.split(":", 1)[1], set()).add(entry["split"])
    for asset in assets:
        destination = OUT / "images" / f"{asset}.jpg"
        source = fetch(asset, destination)
        if source is None:
            failed.append(asset)
            continue
        splits = asset_splits.get(asset, set())
        if len(splits) != 1:
            raise RuntimeError(f"{asset} 在 v2 中对应多个 split: {splits}")
        split = next(iter(splits))
        split_counts[split] += 1
        manifest.append({
            "id": f"benign_coco/{asset}",
            "pair_id": f"benign_coco/{asset}",
            "group": f"coco:{asset}",
            "role": "clean",
            "domain": "image",
            "category": "benign_coco",
            "dataset": "benign_coco",
            "split": split,
            "label": 0,
            "image": str(destination),
            "question": "Describe this image.",
            "source": source,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("splits", split_counts, "failed", len(failed), failed[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
