"""Build the v2 training set: RI text pairs + VisInject pixel pairs.

- text: all 500 RI complete pairs, question "Describe this image."
- pixel: VisInject 4 clean backgrounds x 21 experiments = 84 pairs
- split by background group; own 50 pixel pairs stay as external test
"""
from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(r"D:\CodexProjects\PrivaSee")
RI = ROOT / "data/datasets/ri_multimodal_injection"
OUT = ROOT / "data/datasets/rcs_train_v2"
HF = "https://huggingface.co/datasets/jeffliulab/visinject/resolve/main/"
GH = "https://raw.githubusercontent.com/jeffliulab/vis-inject/main/"
CLEANS = ["bill", "cat", "chat", "code", "dog", "kpop", "webpage"]
EXPERIMENTS = [f"exp_{name}_{config}" for name in
               ("ad", "apple", "card", "email", "news", "obey", "url") for config in ("2m", "3m", "4m")]
QUESTION = "Describe this image."
PIXEL_QUESTIONS = [
    "Describe this image.", "What is shown in this image?", "What can you see in the picture?",
    "Summarize the content of the image.", "What is the main subject?", "Describe the scene.",
    "What objects are in the image?", "Tell me about this picture.", "What is happening here?",
    "What does the image show?", "List the main elements.", "Describe what you see.",
    "What is in the picture?", "Explain the image content.", "What is the focus of this image?",
    "Describe the visual content.", "What details are visible?", "What is depicted here?",
    "Summarize what is visible.", "What is this an image of?", "Describe the main elements.",
]


def download(relative: str, destination: Path, retries: int = 4, base: str = HF) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(base + relative, timeout=90).read()
            destination.write_bytes(data)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{relative}: {last}")


def text_pairs() -> list[dict]:
    rows = list(csv.DictReader((RI / "index.csv").open(encoding="utf-8-sig")))
    manifest = [json.loads(line) for line in (RI / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    background = {row["media_path"]: row.get("source_asset_id") for row in manifest}
    dataset_group = {}
    for family in ("visible_command", "authority_impersonation"):
        dataset_group[family] = "text_visible"
    for family in ("low_contrast", "typographic_obfuscation"):
        dataset_group[family] = "text_obfuscated"
    pairs = []
    for row in rows:
        if row["status"] != "complete":
            continue
        pairs.append({
            "pair_id": f"text/{row['pair_id']}",
            "domain": "text",
            "category": row["family"],
            "dataset": dataset_group.get(row["family"], "text_carrier"),
            "attack": str((RI / "images" / row["attack_path"]).resolve()),
            "clean": str((RI / "images" / row["benign_path"]).resolve()),
            "group": f"coco:{background.get(row['attack_path'], row['pair_id'])}",
            "question": QUESTION,
        })
    return pairs


def pixel_pairs() -> list[dict]:
    pairs = []
    for name in CLEANS:
        clean_src = OUT / f"pixel/_clean/{name}.png"
        download(f"data/images/ORIGIN_{name}.png", clean_src, base=GH)
        with Image.open(clean_src) as image:
            resized = image.convert("RGB").resize((224, 224), Image.LANCZOS)
        clean = OUT / f"pixel/{name}/clean.png"
        clean.parent.mkdir(parents=True, exist_ok=True)
        resized.save(clean)
        for experiment_index, experiment in enumerate(EXPERIMENTS):
            relative = f"experiments/{experiment}/adversarial/adv_ORIGIN_{name}.png"
            attack = OUT / f"pixel/{name}/{experiment}_adv.png"
            download(relative, attack)
            pairs.append({
                "pair_id": f"pixel/visinject/{name}/{experiment}",
                "domain": "pixel",
                "category": "visinject",
                "dataset": "pixel_visinject",
                "attack": str(attack.resolve()),
                "clean": str(clean.resolve()),
                "group": f"visinject:{name}",
                "question": PIXEL_QUESTIONS[experiment_index % len(PIXEL_QUESTIONS)],
            })
    return pairs


def assign_splits(pairs: list[dict], train_ratio=0.8, val_ratio=0.1, seed=42) -> dict[str, str]:
    groups = sorted({p["group"] for p in pairs})
    import random
    random.Random(seed).shuffle(groups)
    n = len(groups)
    mapping = {}
    for index, group in enumerate(groups):
        mapping[group] = "train" if index < round(n * train_ratio) else ("val" if index < round(n * (train_ratio + val_ratio)) else "test")
    return mapping


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text = text_pairs()
    pixel = pixel_pairs()
    split_text = assign_splits(text)
    split_pixel = assign_splits(pixel, train_ratio=0.75, val_ratio=0.25)  # 3 train / 1 val of 4 backgrounds
    entries = []
    for pair in text + pixel:
        split = (split_text if pair["domain"] == "text" else split_pixel)[pair["group"]]
        for role, label in (("attack", 1), ("clean", 0)):
            entries.append({
                "id": f"{pair['pair_id']}/{role}",
                "image": pair["attack"] if role == "attack" else pair["clean"],
                "question": pair["question"],
                "label": label,
                "dataset": pair["dataset"],
                "split": split,
                "pair_id": pair["pair_id"],
                "group": pair["group"],
                "domain": pair["domain"],
                "category": pair["category"],
            })
    (OUT / "manifest.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "pairs.json").write_text(json.dumps(text + pixel, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pairs:", len(text) + len(pixel), "entries:", len(entries))
    print("split counts:", Counter(e["split"] for e in entries))
    print("dataset counts:", Counter(e["dataset"] for e in entries))
    print("text groups:", len({p["group"] for p in text}), "pixel groups:", len({p["group"] for p in pixel}))
    for split in ("train", "val", "test"):
        labels = Counter(e["label"] for e in entries if e["split"] == split)
        print(split, "labels", dict(labels), "datasets", len({e["dataset"] for e in entries if e["split"] == split}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
