"""RCS Qwen input features: one forward per row, atomic per-row caches.

Method: upstream feature_extractor_qwen.py at commit
3ea8e29f3542e7962a7eaed7996dc5654b4035fd (snapshot in third_party/rcs).
Adaptations: existing NF4 loader, PIL Windows paths, mask-aware last token,
FP32 full vectors, explicit nullable labels, content-checked resumable manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from test_qwen_vl import DEFAULT_MODEL_DIR, PROJECT_ROOT, save_results, status

os.environ.setdefault("TMP", str(PROJECT_ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(PROJECT_ROOT / ".tmp"))

UPSTREAM_COMMIT = "3ea8e29f3542e7962a7eaed7996dc5654b4035fd"
CONVENTION = "hidden_states[0]=embedding; 1..36=decoder; 36 includes final RMSNorm"


def output_on_d(path):
    path = Path(path).expanduser().resolve()
    if path.drive.lower() != "d:":
        raise ValueError("RCS 输出必须位于 D 盘")
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_layers(value):
    layers = list(range(37)) if value == "all" else sorted({int(v.strip()) for v in value.split(",")})
    if not layers or any(v < 0 or v > 36 for v in layers):
        raise ValueError("层号必须为0..36，或 all")
    return layers


def read_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("清单必须是非空 JSON 数组")
    result, ids = [], set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("每条记录必须是对象")
        row = dict(raw)
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("每条记录需要唯一、非空字符串 id")
        ids.add(row["id"])
        for key in ("image", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"记录 {row['id']} 缺少 {key}")
        row.setdefault("label", None)
        if row["label"] is not None and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise ValueError("label 只能为0、1或null，未标注不能默认为良性")
        row.setdefault("dataset", None)
        if row["dataset"] is not None and (not isinstance(row["dataset"], str) or not row["dataset"].strip()):
            raise ValueError("dataset 必须为非空字符串或null")
        row.setdefault("split", "unassigned")
        if row["split"] not in ("train", "val", "test", "unassigned"):
            raise ValueError("split 必须为 train/val/test/unassigned")
        image = Path(row["image"]).expanduser()
        row["image"] = str((path.parent / image).resolve() if not image.is_absolute() else image.resolve())
        result.append(row)
    return result


def save_feature(path, vectors, metadata):
    import numpy as np
    path = output_on_d(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, vectors=np.asarray(vectors, dtype=np.float32),
                            metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
    temporary.replace(path)


def load_feature(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        meta = json.loads(str(data["metadata"].item()))
    if vectors.dtype != np.float32 or vectors.shape != (len(meta["layers"]), 2048) or not np.isfinite(vectors).all():
        raise ValueError(f"特征形状、类型或数值错误：{path}")
    return vectors, meta


class QwenHiddenStateExtractor:
    """Same input-only forward/extraction as RCS, with PrivaSee NF4 loading."""
    def __init__(self, min_visual_tokens=256, max_visual_tokens=768):
        self.min_visual_tokens = min_visual_tokens
        self.max_visual_tokens = max_visual_tokens
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("提取使用现有 CUDA/NF4 模型，需要可用 CUDA")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULT_MODEL_DIR, local_files_only=True, device_map="auto", low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        model.eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_DIR, local_files_only=True, use_fast=False,
            min_pixels=self.min_visual_tokens * 28 * 28, max_pixels=self.max_visual_tokens * 28 * 28)
        self.model, self.processor = model, processor

    def extract(self, row, layers):
        import torch
        from PIL import Image, ImageOps
        from qwen_vl_utils import process_vision_info
        from demo_rcs_hidden_state import last_valid_token_index
        self.load()
        with Image.open(row["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": row["question"]}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True,
                                return_tensors="pt").to(self.model.device)
        index = last_valid_token_index(inputs.attention_mask)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            vectors = torch.stack([outputs.hidden_states[layer][0, index].float().cpu() for layer in layers]).numpy()
            del outputs
        if not __import__("numpy").isfinite(vectors).all():
            raise RuntimeError("隐藏向量包含NaN/Inf")
        token_id = int(inputs.input_ids[0, index].item())
        details = {"token_index": index, "token_id": token_id,
                   "token_text": self.processor.tokenizer.decode([token_id], skip_special_tokens=False),
                   "input_token_count": int(inputs.attention_mask.sum().item()),
                   "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "input_ids": inputs.input_ids[0].cpu().tolist(),
                   "image_grid_thw": inputs.image_grid_thw.cpu().tolist()}
        return vectors, details


def run_batch(manifest, output, layers, min_visual_tokens=256, max_visual_tokens=768,
              extractor=None, validate_only=False):
    rows = read_manifest(manifest)
    output = output_on_d(output)
    if not layers or layers != sorted(set(layers)) or any(type(v) is not int or not 0 <= v <= 36 for v in layers):
        raise ValueError("layers 必须是升序不重复的0..36整数列表")
    if not 4 <= min_visual_tokens <= max_visual_tokens:
        raise ValueError("视觉token预算必须满足 4 <= min <= max")
    config = {"schema": 1, "upstream_commit": UPSTREAM_COMMIT, "model": str(DEFAULT_MODEL_DIR.resolve()),
              "model_config_hash": file_hash(DEFAULT_MODEL_DIR / "config.json"),
              "processor_hashes": {p.name: file_hash(p) for p in DEFAULT_MODEL_DIR.iterdir()
                                   if p.is_file() and p.suffix == ".json" and "model.safetensors" not in p.name},
              "layers": layers, "layer_convention": CONVENTION, "dimension": 2048,
              "min_visual_tokens": min_visual_tokens, "max_visual_tokens": max_visual_tokens,
              "quantization": "NF4 double quantization; BF16 compute; FP32 saved",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "qwen-vl-utils")}}
    config["fingerprint"] = digest(config)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("输出目录配置不同，请使用新目录；不能混用层号、预处理或运行环境")
    if validate_only:
        for row in rows:
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
        status(f"{len(rows)}条记录预检通过；层号 {layers}；没有加载模型或写入结果")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    save_results(config_path, config)
    extractor = extractor or QwenHiddenStateExtractor(min_visual_tokens, max_visual_tokens)
    records, failures = [], 0
    for row in rows:
        key = hashlib.sha256(row["id"].encode()).hexdigest()
        relative = f"features/{key}.npz"
        feature_path = output / relative
        entry = {"id": row["id"], "row": row, "file": relative}
        try:
            image_hash = file_hash(row["image"])
            signature = digest({"row": row, "image_hash": image_hash, "config": config["fingerprint"]})
            reusable = False
            if feature_path.exists():
                try:
                    _, previous = load_feature(feature_path)
                    reusable = previous.get("signature") == signature
                except Exception:
                    # Invalid ZIP/JSON, incomplete writes and stale schemas are
                    # cache misses, not permanent failures of the source input.
                    pass
            if reusable:
                status(f"跳过已完成：{row['id']}")
            else:
                status(f"提取：{row['id']}，层号 {layers}")
                vectors, details = extractor.extract(row, layers)
                meta = {"row": row, "image_sha256": image_hash, "layers": layers,
                        "signature": signature, "config_fingerprint": config["fingerprint"], **details}
                save_feature(feature_path, vectors, meta)
                load_feature(feature_path)
            entry.update(status="completed", signature=signature)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures += 1
            status(f"失败后继续：{row['id']}：{exc}")
        records.append(entry)
        save_results(output / "index.json", {"schema": 1, "manifest": str(Path(manifest).resolve()),
                     "config_fingerprint": config["fingerprint"], "records": records})
    status(f"完成：{len(records)-failures}条成功／跳过，{failures}条失败；{output}")
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="12,18,24")
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    return run_batch(args.manifest, args.output, parse_layers(args.layers), args.min_visual_tokens,
                     args.max_visual_tokens, validate_only=args.validate_only)


if __name__ == "__main__":
    raise SystemExit(main())
