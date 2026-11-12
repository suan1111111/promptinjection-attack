"""Projection training variant with an explicit anti-collapse term.

Keeps the paper's network and dataset-separation + class-separation losses,
but (1) drops the intra-class compactness term (paper Eq. 3 has no such term)
and (2) adds a small variance/covariance regulariser so the 256-d output does
not collapse onto a single direction. Selects the checkpoint by validation
AUROC instead of the last epoch.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from rcs_detector import KCDDetector, load_layer
from rcs_projection import project_vectors, save_projection, training_arrays
from rcs_projection_core import LearnedProjection
from rcs_training_utils import ensure_full_batch

PAPER_SEP_WEIGHT = 5.0


def dataset_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    unique = torch.unique(labels)
    if len(unique) < 2:
        return torch.zeros((), device=z.device)
    distances = torch.cdist(z, z, p=2)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff = 1.0 - same
    eye = 1.0 - torch.eye(len(labels), device=z.device)
    same = same * eye
    diff = diff * eye
    intra = (distances * same).sum() / same.sum() if same.sum() > 0 else torch.zeros((), device=z.device)
    inter = (torch.clamp(margin - distances, min=0) * diff).sum() / diff.sum() if diff.sum() > 0 else torch.zeros((), device=z.device)
    return intra + inter


def separation_loss(z: torch.Tensor, y: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    benign, malicious = z[y == 0], z[y == 1]
    if len(benign) == 0 or len(malicious) == 0:
        return torch.zeros((), device=z.device)
    centroids = torch.stack([benign.mean(dim=0), malicious.mean(dim=0)])
    distance = F.pairwise_distance(centroids[0].unsqueeze(0), centroids[1].unsqueeze(0))
    return torch.clamp(margin - distance, min=0).mean()


def variance_loss(z: torch.Tensor, target_std: float = 0.1) -> torch.Tensor:
    std = z.std(dim=0)
    hinge = torch.relu(target_std - std).mean()
    covariance = (z.T @ z) / len(z)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return hinge + 0.1 * off_diagonal.pow(2).mean()


def rank_share(z: np.ndarray) -> float:
    centered = z - z.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    return float(energy[0] / energy.sum()) if energy.sum() else 1.0


def validation_auc(model, train_x, train_y, val_x, val_y) -> float:
    train_z = project_vectors(model, train_x)
    val_z = project_vectors(model, val_x)
    detector = KCDDetector(k=15).fit(train_z, train_y)
    return float(roc_auc_score(val_y, detector.score(val_z)))


def train(feature_dir: Path, output: Path, layer: int, epochs: int = 100, batch_size: int = 64,
          learning_rate: float = 1e-3, variance_weight: float = 0.5, device: str = "cpu",
          seed: int = 42, patience: int = 15, sep_weight: float = 5.0, sep_margin: float = 2.0,
          dataset_weight: float = 1.0) -> dict:
    features, source_labels, safety_labels, mapping, config, _ = training_arrays(feature_dir, layer)
    ensure_full_batch(len(features), batch_size)
    split = load_layer(feature_dir, layer)
    val_x, val_y = split["val"]["x"], split["val"]["y"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    architecture = {"input_dim": int(features.shape[1]), "output_dim": 256, "hidden_dim": 512, "dropout": 0.3}
    model = LearnedProjection(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(source_labels), torch.from_numpy(safety_labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))

    history = []
    rank_history = []
    best_auc, best_epoch, best_state = -1.0, 0, None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, d, y in loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = F.normalize(model(x), p=2, dim=1)
            loss = (dataset_weight * dataset_loss(z, d)
                    + sep_weight * separation_loss(z, y, margin=sep_margin)
                    + variance_weight * variance_loss(z))
            if not torch.isfinite(loss):
                raise RuntimeError("loss 不是有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach().item())
        scheduler.step()
        auc = validation_auc(model, features, safety_labels, val_x, val_y)
        share = rank_share(project_vectors(model, features))
        history.append({"epoch": epoch, "loss": total / len(loader), "val_auroc": auc, "rank_share": share})
        if epoch % 10 == 0:
            print(f"layer={layer} epoch={epoch}/{epochs} loss={total/len(loader):.4f} "
                  f"val_auc={auc:.4f} rank_share={share:.3f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"layer={layer} early stop at {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "schema": 1,
        "layer": layer,
        "architecture": architecture,
        "dataset_mapping": mapping,
        "training_sample_count": int(len(features)),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "variance_weight": variance_weight,
        "sep_weight": sep_weight,
        "sep_margin": sep_margin,
        "dataset_weight": dataset_weight,
        "compactness": "removed (paper Eq.3 only)",
        "selection": "best validation AUROC",
        "history": history,
        "feature_config_fingerprint": config["fingerprint"],
    }
    save_projection(output, model, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="8,9,13,14")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--variance-weight", type=float, default=0.5)
    parser.add_argument("--sep-weight", type=float, default=5.0)
    parser.add_argument("--sep-margin", type=float, default=2.0)
    parser.add_argument("--dataset-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for layer in [int(v) for v in args.layers.split(",")]:
        path = args.output / f"layer_{layer}.pt"
        if path.exists():
            raise ValueError(f"{path} 已存在")
    for layer in [int(v) for v in args.layers.split(",")]:
        meta = train(args.features, args.output / f"layer_{layer}.pt", layer, args.epochs,
                     args.batch_size, args.learning_rate, args.variance_weight, args.device, args.seed,
                     sep_weight=args.sep_weight, sep_margin=args.sep_margin, dataset_weight=args.dataset_weight)
        print(f"layer {layer}: best_epoch={meta['best_epoch']} best_val_auroc={meta['best_val_auroc']:.4f} "
              f"final_rank_share={meta['history'][-1]['rank_share']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
