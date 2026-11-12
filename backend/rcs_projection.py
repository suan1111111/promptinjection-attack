"""Train/load a layer-specific RCS projection from cached Qwen features.

Network and loss are verbatim upstream definitions in rcs_projection_core.py.
Optimizer/scheduler/clip/drop_last/early stopping follow train_learned_projection
in the pinned kcd.py. Adds strict feature/label checks and portable checkpoints.
This module does not load a VLM or train a detector/threshold.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rcs_features import UPSTREAM_COMMIT, digest, load_feature, output_on_d, parse_layers
from rcs_projection_core import CONFIG, LearnedProjection, compute_contrastive_loss
from test_qwen_vl import save_results, status


def training_arrays(feature_dir, layer):
    feature_dir = Path(feature_dir).resolve()
    config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((feature_dir / "index.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in config.items() if k != "fingerprint"}) != config["fingerprint"]:
        raise ValueError("缓存配置指纹校验失败")
    if index["config_fingerprint"] != config["fingerprint"] or layer not in config["layers"]:
        raise ValueError("缓存配置不匹配或缺少所选层")
    vectors, labels, datasets, identities, seen = [], [], [], [], set()
    for record in index["records"]:
        row = record["row"]
        if row["split"] != "train":
            continue
        if record["status"] != "completed":
            raise ValueError(f"训练记录提取失败：{row['id']}")
        if row["label"] is None or row["dataset"] is None:
            raise ValueError(f"训练记录缺少安全标签或来源标签：{row['id']}")
        if type(row["label"]) is not int or row["label"] not in (0, 1) or not isinstance(row["dataset"], str) or not row["dataset"].strip():
            raise ValueError("训练记录标签不合法")
        path = (feature_dir / record["file"]).resolve()
        if not path.is_relative_to(feature_dir):
            raise ValueError("特征文件路径越出缓存目录")
        features, meta = load_feature(path)
        if meta["signature"] != record["signature"] or meta["row"] != row or meta["config_fingerprint"] != config["fingerprint"]:
            raise ValueError("特征与清单不一致，请重新提取")
        identity = digest({"image": meta["image_sha256"], "question": row["question"]})
        if identity in seen:
            raise ValueError("训练集中存在重复图片内容与问题，不能重复计数")
        seen.add(identity)
        vectors.append(features[meta["layers"].index(layer)])
        labels.append(row["label"])
        datasets.append(row["dataset"])
        identities.append(identity)
    if len(vectors) < 2 or set(labels) != {0, 1}:
        raise ValueError("投影训练需要至少两个样本，并同时包含良性和恶意标签")
    if len(set(datasets)) < 2:
        raise ValueError("作者的来源结构损失需要至少两个 dataset 来源标签")
    # Protect train/val/test isolation by image content + question, not only IDs.
    for record in index["records"]:
        row = record["row"]
        if row["split"] in ("val", "test") and record["status"] == "completed":
            path = (feature_dir / record["file"]).resolve()
            if not path.is_relative_to(feature_dir):
                raise ValueError("特征路径越界")
            _, meta = load_feature(path)
            if digest({"image": meta["image_sha256"], "question": row["question"]}) in seen:
                raise ValueError("训练集与验证／测试集存在相同图片问题，需先修正划分")
    mapping = {name: i for i, name in enumerate(sorted(set(datasets)))}
    return (np.stack(vectors), np.array([mapping[d] for d in datasets]),
            np.array(labels), mapping, config, identities)


def save_projection(path, model, metadata):
    path = output_on_d(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save({"schema": 1, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, temp)
    temp.replace(path)


def load_projection(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != 1:
        raise ValueError("不支持的投影checkpoint版本")
    metadata = checkpoint["metadata"]
    model = LearnedProjection(**metadata["architecture"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, metadata


def project_vectors(model, vectors, device=None, batch_size=256):
    """Return RAW projection outputs. KCD normalization is a later detector step."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != model.projection[0].in_features or not np.isfinite(vectors).all():
        raise ValueError("输入必须是有限的 N×2048 特征矩阵")
    if batch_size < 1:
        raise ValueError("batch_size 必须为正")
    device = device or next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(vectors), batch_size):
            chunks.append(model(torch.from_numpy(vectors[start:start+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, model.projection[-1].num_features), dtype=np.float32)


def train_projection(feature_dir, output, layer, epochs=100, batch_size=64, learning_rate=1e-3,
                     device="cpu", seed=42):
    output = output_on_d(output)
    if epochs < 1 or batch_size < 2 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs>0、batch_size>=2、learning_rate>0")
    features, source_labels, safety_labels, mapping, config, identities = training_arrays(feature_dir, layer)
    if len(features) < batch_size:
        raise ValueError(f"只有{len(features)}条训练数据，小于batch_size={batch_size}；请显式调小批大小（至少2）")
    if output.exists():
        raise ValueError(f"投影文件已存在，避免覆盖已有训练：{output}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(source_labels).long(),
                            torch.from_numpy(safety_labels).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    architecture = {"input_dim": 2048, "output_dim": 256, "hidden_dim": 512, "dropout": 0.3}
    model = LearnedProjection(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                         eta_min=learning_rate * 0.05)
    best, patience, history, best_epoch = float("inf"), 0, [], 0
    for epoch in range(1, epochs+1):
        model.train()
        totals = np.zeros(3)
        for x, d, y in loader:
            optimizer.zero_grad()
            losses = compute_contrastive_loss(model(x.to(device)), d.to(device), y.to(device))
            if not torch.isfinite(losses[0]).item():
                raise RuntimeError("投影loss包含NaN/Inf")
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            totals += [float(v.detach().item()) for v in losses]
        scheduler.step()
        avg = totals / len(loader)
        history.append({"epoch": epoch, "loss": float(avg[0]), "dataset_loss": float(avg[1]),
                        "toxicity_loss": float(avg[2]), "lr": optimizer.param_groups[0]["lr"]})
        status(f"layer={layer} epoch={epoch}/{epochs} loss={avg[0]:.6f}")
        if avg[0] < best:
            best, best_epoch, patience = float(avg[0]), epoch, 0
        else:
            patience += 1
        if patience >= CONFIG.PROJECTION_MAX_PATIENCE:
            break
    model.eval()
    metadata = {"upstream_commit": UPSTREAM_COMMIT, "layer": layer, "architecture": architecture,
                "feature_config": config, "dataset_mapping": mapping, "training_sample_count": len(features),
                "training_identity_hash": digest(identities), "epochs_requested": epochs,
                "batch_size": batch_size, "seed": seed, "learning_rate": learning_rate,
                "optimizer": "Adam weight_decay=1e-4; cosine eta_min=0.05*lr; clip_norm=1; drop_last=True",
                "loss_weights": {"dataset": 1.0, "toxicity": 5.0}, "history": history,
                "best_epoch": best_epoch,
                "selection": "last epoch, matching upstream; early stop monitors training loss, not detection validation"}
    save_projection(output, model, metadata)
    save_results(output.with_suffix(".json"), metadata)
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="新输出目录，每层一个独立投影")
    parser.add_argument("--layers", default="18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = output_on_d(args.output)
    layers = parse_layers(args.layers)
    # Preflight ALL layers before starting training any layer.
    for layer in layers:
        values = training_arrays(args.features, layer)
        if len(values[0]) < args.batch_size or args.batch_size < 2:
            raise ValueError("训练样本数不足以组成指定批大小")
        if (output / f"layer_{layer}.pt").exists():
            raise ValueError("输出层文件已存在，请指定新目录")
    for layer in layers:
        train_projection(args.features, output / f"layer_{layer}.pt", layer, args.epochs,
                         args.batch_size, args.learning_rate, args.device, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
