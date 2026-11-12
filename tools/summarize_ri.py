"""Summarize the downloaded RI image set and write a per-pair index CSV."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(r"D:\CodexProjects\PrivaSee\data\datasets\ri_multimodal_injection")
rows = [json.loads(line) for line in (BASE / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

pairs: dict[str, dict[str, dict]] = defaultdict(dict)
for row in rows:
    pairs[row["pair_id"]][row["media_path"]] = row


def classify(group: dict[str, dict]) -> tuple[str, str | None, str | None]:
    attacks = [r for r in group.values() if r["is_attack"]]
    benigns = [r for r in group.values() if not r["is_attack"]]
    attack = attacks[0]["media_path"] if attacks else None
    benign = benigns[0]["media_path"] if benigns else None
    if attacks and benigns:
        status = "complete"
    elif attacks:
        status = "attack_only"
    elif benigns:
        status = "benign_only"
    else:
        status = "unknown"
    return status, attack, benign


def family_of(group: dict[str, dict]) -> str:
    for row in group.values():
        return row.get("attack_family") or "?"
    return "?"


with (BASE / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["pair_id", "family", "split", "status", "attack_path", "benign_path", "attack_text", "benign_text"])
    for pair_id, group in sorted(pairs.items()):
        status, attack, benign = classify(group)
        any_row = next(iter(group.values()))
        atk_text = next((r["media_text"] for r in group.values() if r["is_attack"]), "")
        ben_text = next((r["media_text"] for r in group.values() if not r["is_attack"]), "")
        writer.writerow([pair_id, family_of(group), any_row["split"], status, attack or "", benign or "", atk_text or "", ben_text or ""])

summary: dict[str, dict[str, int]] = defaultdict(lambda: {"attack": 0, "benign": 0, "complete": 0, "attack_only": 0, "benign_only": 0})
for pair_id, group in pairs.items():
    status, _, _ = classify(group)
    family = family_of(group)
    entry = summary[family]
    entry["attack"] += sum(1 for r in group.values() if r["is_attack"])
    entry["benign"] += sum(1 for r in group.values() if not r["is_attack"])
    entry[status] += 1

print("family".ljust(32), "attack".rjust(7), "benign".rjust(7), "complete".rjust(9), "attack_only".rjust(12), "benign_only".rjust(12))
for family in sorted(summary):
    e = summary[family]
    print(family.ljust(32), str(e["attack"]).rjust(7), str(e["benign"]).rjust(7), str(e["complete"]).rjust(9),
          str(e["attack_only"]).rjust(12), str(e["benign_only"]).rjust(12))
print("\ntotal rows:", len(rows), "| pairs:", len(pairs), "| complete:", sum(1 for g in pairs.values() if classify(g)[0] == "complete"))
print("index written:", BASE / "index.csv")
