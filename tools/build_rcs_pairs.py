"""Build the 100-pair (200 image) RCS dataset.

- text side : 50 RI pairs, 5 per family, balanced
- pixel side: 50 own perturbation pairs, 10 per perturbation type
Everything is copied under data/datasets/rcs_pairs_100/.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(r"D:\CodexProjects\PrivaSee")
RI = ROOT / "data/datasets/ri_multimodal_injection"
OUT = ROOT / "data/datasets/rcs_pairs_100"
TEXT_DIR = OUT / "text"
PIXEL_DIR = OUT / "pixel"

TEXT_FAMILIES = [
    "visible_command", "low_contrast", "typographic_obfuscation", "small_corner_text",
    "cross_region_completion", "reachable_qr", "quotation_ambiguity",
    "authority_impersonation", "task_relevant_disguise", "reachable_metadata",
]
PER_FAMILY = 5

BATCHES = {
    "region": "scene_region_top10_20261103",
    "global": "scene_global_top10_20261104",
    "border": "scene_border_top10_20261105",
    "fixed": "scene_fixed_top10_20261106",
    "flower": "scene_flower_top10_20261107",
}


def save_png(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image.convert("RGB").save(dst, format="PNG")


def pick_even(items: list, k: int) -> list:
    items = sorted(items)
    if len(items) <= k:
        return items
    if k == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def build_text() -> list[dict]:
    rows = list(csv.DictReader((RI / "index.csv").open(encoding="utf-8-sig")))
    manifest = [json.loads(line) for line in (RI / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    background = {row["media_path"]: row.get("source_asset_id") for row in manifest}
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete" and row["family"] in TEXT_FAMILIES:
            by_family[row["family"]].append(row)

    pairs = []
    for family in TEXT_FAMILIES:
        chosen = pick_even([r["pair_id"] for r in by_family[family]], PER_FAMILY)
        index = {r["pair_id"]: r for r in by_family[family]}
        for pair_id in chosen:
            row = index[pair_id]
            folder = TEXT_DIR / family
            folder.mkdir(parents=True, exist_ok=True)
            attack = folder / f"{pair_id}_a{Path(row['attack_path']).suffix}"
            clean = folder / f"{pair_id}_b{Path(row['benign_path']).suffix}"
            save_png(RI / "images" / row["attack_path"], attack)
            save_png(RI / "images" / row["benign_path"], clean)
            pairs.append({
                "pair_id": f"text/{pair_id}",
                "domain": "text",
                "category": family,
                "attack": str(attack.relative_to(OUT)).replace("\\", "/"),
                "clean": str(clean.relative_to(OUT)).replace("\\", "/"),
                "attack_text": row["attack_text"],
                "clean_text": row["benign_text"],
                "group": f"coco:{background.get(row['attack_path'], pair_id)}",
                "background_id": background.get(row["attack_path"]),
                "split": row["split"],
                "source": str((RI / "images" / row["attack_path"]).relative_to(ROOT)).replace("\\", "/"),
            })
    return pairs


def build_pixel() -> list[dict]:
    pairs = []
    for kind, batch in BATCHES.items():
        status_path = ROOT / "outputs" / batch / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        for job_id, entry in status["jobs"].items():
            if entry.get("state") != "effective":
                continue
            job_dir = Path(entry["directory"])
            clean = job_dir / "run" / "clean_resized.png"
            attack = Path(entry["image"]) if entry.get("image") else None
            if attack is None or not attack.is_file():
                matches = list((ROOT / "outputs" / batch / "effective_images").glob(f"*_{job_id}.png"))
                attack = matches[0] if matches else job_dir / "run" / "adversarial_final.png"
            if not clean.is_file() or not attack.is_file():
                print(f"  skip {kind}/{job_id}: missing file")
                continue
            folder = PIXEL_DIR / kind
            folder.mkdir(parents=True, exist_ok=True)
            attack_out = folder / f"{kind}_{job_id}_adv.png"
            clean_out = folder / f"{kind}_{job_id}_clean.png"
            shutil.copy2(attack, attack_out)
            shutil.copy2(clean, clean_out)
            job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            pairs.append({
                "pair_id": f"pixel/{kind}/{job_id}",
                "domain": "pixel",
                "category": kind,
                "attack": str(attack_out.relative_to(OUT)).replace("\\", "/"),
                "clean": str(clean_out.relative_to(OUT)).replace("\\", "/"),
                "attack_text": None,
                "clean_text": None,
                "group": f"orig:{job_id}",
                "background_id": None,
                "split": None,
                "source": str(Path(job["image"]).relative_to(ROOT)).replace("\\", "/"),
            })
    return pairs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text_pairs = build_text()
    pixel_pairs = build_pixel()
    pairs = text_pairs + pixel_pairs
    (OUT / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "domain", "category", "attack", "clean", "group", "split", "source"])
        for pair in pairs:
            writer.writerow([pair["pair_id"], pair["domain"], pair["category"], pair["attack"],
                             pair["clean"], pair["group"], pair["split"] or "", pair["source"]])
    counts = Counter(p["category"] for p in pairs)
    print("total pairs:", len(pairs))
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print("text:", len(text_pairs), "pixel:", len(pixel_pairs))
    print("images:", sum(1 for _ in OUT.rglob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
