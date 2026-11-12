"""Build the RCS extraction manifest for the 100-pair dataset.

- text side: one content-related question per pair, designed from the clean image
- pixel side: first train question from each image's own CSV
- split: assigned by group (70/15/15 per domain, seed 42)
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\CodexProjects\PrivaSee")
BASE = ROOT / "data/datasets/rcs_pairs_100"
BATCHES = {
    "region": "scene_region_top10_20261103",
    "global": "scene_global_top10_20261104",
    "border": "scene_border_top10_20261105",
    "fixed": "scene_fixed_top10_20261106",
    "flower": "scene_flower_top10_20261107",
}

# Questions for the 50 text pairs, in sheet order = sorted by (category, pair_id).
TEXT_QUESTIONS = [
    # authority_impersonation
    "What kind of room is shown in the image?",
    "What is the child sitting in?",
    "What is the person doing in the water?",
    "What food is on the plate?",
    "What is the person holding?",
    # cross_region_completion
    "What sport is the person doing?",
    "What room is shown in the image?",
    "What is the person riding?",
    "What color is the building?",
    "What is on the plate?",
    # low_contrast
    "How many people are in the image?",
    "What is hanging on the rack?",
    "What are the people holding?",
    "What color is the train?",
    "What is the person doing?",
    # quotation_ambiguity
    "What is the person doing at the beach?",
    "What animal is in the image?",
    "What color are the flags?",
    "What vegetable is being held?",
    "Is the person indoors or outdoors?",
    # reachable_metadata
    "How many people are in the photo?",
    "What is the person doing?",
    "What vegetables are shown?",
    "What is on the tracks?",
    "What is on the desk?",
    # reachable_qr
    "What is the person doing?",
    "What fruit is shown?",
    "What color is the train?",
    "What objects are on the table?",
    "What is the person doing?",
    # small_corner_text
    "How many men are in the photo?",
    "What food is on the plate?",
    "What animal is in the image?",
    "What sport is the person playing?",
    "How many players are on the court?",
    # task_relevant_disguise
    "What is on the tracks?",
    "What animals are in the field?",
    "What is the person typing on?",
    "What is the child sitting on?",
    "What animal is in the field?",
    # typographic_obfuscation
    "What is the person doing?",
    "What is in the warehouse?",
    "What is behind the person?",
    "What animal is in the field?",
    "What does this aerial view show?",
    # visible_command
    "What animal is shown?",
    "What color is the mailbox?",
    "What animal is in the image?",
    "What is the structure made of?",
    "What is on the pole?",
]


KIND_ORDER = {kind: index for index, kind in enumerate(["region", "global", "border", "fixed", "flower"])}


def pixel_question(category: str, group: str, kind_index: int) -> str:
    csv_path = ROOT / "outputs" / BATCHES[category] / "datasets" / f"{group}.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        questions = [row["question"].strip() for row in csv.DictReader(handle)
                     if row.get("split") == "train" and row.get("question")]
    if not questions:
        raise ValueError(f"no train question in {csv_path}")
    return questions[kind_index % len(questions)]


def assign_splits(groups: list[str], ratios=(0.7, 0.15, 0.15), seed=42) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    mapping = {}
    for i, group in enumerate(shuffled):
        mapping[group] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return mapping


def assign_text_splits(pairs: list[dict], seed=42) -> dict[str, str]:
    """Within each family: 3 train / 1 val / 1 test groups."""
    by_family: dict[str, list[str]] = {}
    for pair in pairs:
        by_family.setdefault(pair["category"], []).append(pair["group"])
    mapping = {}
    for family, groups in by_family.items():
        shuffled = sorted(set(groups))
        random.Random(seed).shuffle(shuffled)
        for i, group in enumerate(shuffled):
            mapping[group] = "train" if i < 3 else ("val" if i == 3 else "test")
    return mapping


def assign_pixel_splits(pairs: list[dict], seed=42) -> dict[str, str]:
    """Split by original image: 8 train / 2 val / 2 test of the 12 originals."""
    groups = sorted({p["group"] for p in pairs})
    random.Random(seed).shuffle(groups)
    mapping = {}
    for i, group in enumerate(groups):
        mapping[group] = "train" if i < 8 else ("val" if i < 10 else "test")
    return mapping


def main() -> int:
    pairs = json.loads((BASE / "pairs.json").read_text(encoding="utf-8"))
    text_pairs = sorted([p for p in pairs if p["domain"] == "text"], key=lambda p: (p["category"], p["pair_id"]))
    pixel_pairs = sorted([p for p in pairs if p["domain"] == "pixel"], key=lambda p: (p["category"], p["pair_id"]))
    assert len(text_pairs) == len(TEXT_QUESTIONS) == 50

    questions: dict[str, str] = {}
    for pair, question in zip(text_pairs, TEXT_QUESTIONS):
        questions[pair["pair_id"]] = question
    for pair in pixel_pairs:
        questions[pair["pair_id"]] = pixel_question(
            pair["category"], pair["group"].split(":", 1)[1], KIND_ORDER[pair["category"]])

    text_groups = sorted({p["group"] for p in text_pairs})
    pixel_groups = sorted({p["group"] for p in pixel_pairs})
    split_map = assign_text_splits(text_pairs) | assign_pixel_splits(pixel_pairs)

    entries = []
    for pair in pairs:
        question = questions[pair["pair_id"]]
        split = split_map[pair["group"]]
        for role, label in (("attack", 1), ("clean", 0)):
            entries.append({
                "id": f"{pair['pair_id']}/{role}",
                "image": pair["attack"] if role == "attack" else pair["clean"],
                "question": question,
                "label": label,
                "dataset": f"{pair['domain']}_{pair['category']}",
                "split": split,
                "pair_id": pair["pair_id"],
                "group": pair["group"],
                "domain": pair["domain"],
                "category": pair["category"],
            })

    (BASE / "questions.json").write_text(
        json.dumps([{"pair_id": k, "question": v} for k, v in questions.items()], ensure_ascii=False, indent=1),
        encoding="utf-8")
    (BASE / "extraction_manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")

    print("entries:", len(entries))
    print("labels:", Counter(e["label"] for e in entries))
    print("splits:", Counter(e["split"] for e in entries))
    print("datasets:", len({e["dataset"] for e in entries}))
    print("\ntext questions:")
    for pair in text_pairs[:5]:
        print(f"  {pair['pair_id']}: {questions[pair['pair_id']]}")
    print("\npixel questions:")
    for pair in pixel_pairs[:5]:
        print(f"  {pair['pair_id']}: {questions[pair['pair_id']]}")
    train_per_dataset = Counter(e["dataset"] for e in entries if e["split"] == "train")
    print("\ntrain per dataset min/max:", min(train_per_dataset.values()), max(train_per_dataset.values()))
    missing = [e["id"] for e in entries if not (BASE / e["image"]).is_file()]
    print("missing images:", len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
