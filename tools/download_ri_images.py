"""Download all RI Multimodal Injection image rows into the project folder.

Folder layout (everything on D:):
  data/datasets/ri_multimodal_injection/images/media/image/<family>/<sample>.png
  data/datasets/ri_multimodal_injection/manifest.jsonl
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(r"D:\CodexProjects\PrivaSee\data\datasets\ri_multimodal_injection")
IMAGES = BASE / "images"
MANIFEST = BASE / "manifest.jsonl"
SPLITS = ["calibration", "dev", "external", "hard_benign", "policy", "test", "train"]
DATASET = "hanIshi%2Fri-multimodal-injection"
WORKERS = 6


def get(url: str, retries: int = 6) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"download failed: {url}: {last}")


def fetch_page(split: str, offset: int) -> dict:
    api = (
        f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
        f"&config=image&split={split}&offset={offset}&length=100"
    )
    return json.loads(get(api))


def save_one(item: dict) -> dict | None:
    row = item["row"]
    media_path = row.get("media_path")
    image = row.get("image")
    if not media_path or not image or not image.get("src"):
        return None
    destination = IMAGES / media_path
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = get(image["src"], retries=4)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return {
        "split": item["split"],
        "media_path": media_path,
        "is_attack": row.get("is_attack"),
        "pair_id": row.get("pair_id"),
        "attack_family": row.get("attack_family"),
        "source_asset_id": row.get("source_asset_id"),
        "media_text": row.get("media_text"),
    }


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    existing: set[tuple[str, str]] = set()
    if MANIFEST.is_file():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                existing.add((record["split"], record["media_path"]))
            except Exception:  # noqa: BLE001
                pass
    mode = "a" if MANIFEST.exists() else "w"
    total_saved = 0
    with MANIFEST.open(mode, encoding="utf-8") as manifest:
        for split in SPLITS:
            offset = 0
            while True:
                payload = fetch_page(split, offset)
                rows = payload["rows"]
                total = payload.get("num_rows_total", 0)
                jobs = [{"row": item["row"], "split": split} for item in rows]
                saved = 0
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    futures = {pool.submit(save_one, job): job for job in jobs}
                    for future in as_completed(futures):
                        try:
                            record = future.result()
                        except Exception as exc:  # noqa: BLE001
                            print(f"  failed {futures[future]['row'].get('media_path')}: {exc}", flush=True)
                            continue
                        if record and (record["split"], record["media_path"]) not in existing:
                            existing.add((record["split"], record["media_path"]))
                            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                            manifest.flush()
                            saved += 1
                total_saved += saved
                offset += len(rows)
                print(f"{split}: {offset}/{total} (saved {saved})", flush=True)
                if offset >= total or not rows:
                    break
    count = sum(1 for _ in IMAGES.rglob("*.png"))
    print(f"done: {count} png files, {total_saved} new manifest rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
