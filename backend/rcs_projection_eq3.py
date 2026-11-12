"""Paper Eq.3 projection training (no intra-class compactness term).

Loss: L_dataset (margin 1.0, weight 1.0) + 5.0 * L_sep
      L_sep = hinge(margin_toxicity - centroid distance), margin_toxicity = 2.0
      The released code additionally adds an intra-class compactness term; the
      paper equation does not, and this variant omits it.

Loop matches upstream train_learned_projection:
      100 epochs, batch 64, Adam(1e-3, wd 1e-4), cosine annealing (0.05x min),
      grad clip 1.0, shuffle + drop_last, early stop on average training loss
      with patience 15, save the LAST epoch.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rcs_projection import save_projection, training_arrays
from rcs_projection_core import LearnedProjection
from rcs_training_utils import ensure_full_batch

DATASET_WEIGHT = 1.0
SEP_WEIGHT = 5.0
SEP_MARGIN = 2.0
PATIENCE = 15


def dataset_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    if len(torch.unique(labels)) < 2:
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


def separation_loss(z: torch.Tensor, y: torch.Tensor, margin: float = SEP_MARGIN) -> torch.Tensor:
    benign, malicious = z[y == 0], z[y == 1]
    if len(benign) == 0 or len(malicious) == 0:
        return torch.zeros((), device=z.device)
    centroids = torch.stack([benign.mean(dim=0), malicious.mean(dim=0)])
    distance = F.pairwise_distance(centroids[0].unsqueeze(0), centroids[1].unsqueeze(0))
    return torch.clamp(margin - distance, min=0).mean()


def train(feature_dir: Path, output: Path, layer: int, epochs: int = 100, batch_size: int = 64,
          learning_rate: float = 1e-3, device: str = "cpu", seed: int = 42,
          sep_margin: float = SEP_MARGIN) -> dict:
    features, source_labels, safety_labels, mapping, config, _ = training_arrays(feature_dir, layer)
    ensure_full_batch(len(features), batch_size)

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
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, d, y in loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = F.normalize(model(x), p=2, dim=1)
            loss = DATASET_WEIGHT * dataset_loss(z, d) + SEP_WEIGHT * separation_loss(z, y, margin=sep_margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item())
        scheduler.step()
        average = total / len(loader)
        history.append({"epoch": epoch, "loss": average})
        if epoch % 10 == 0:
            print(f"layer={layer} epoch={epoch}/{epochs} loss={average:.4f}", flush=True)
        if average < best_loss:
            best_loss, best_epoch, stale = average, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"layer={layer} early stop at {epoch}", flush=True)
                break

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
        "selection": "last epoch, matching upstream; early stop monitors training loss, not detection validation",
        "loss": f"paper Eq.3: L_dataset(1.0, margin 1.0) + 5.0 * centroid hinge(margin {sep_margin}); no compactness",
        "sep_margin": sep_margin,
        "history": history,
        "feature_config_fingerprint": config["fingerprint"],
    }
    save_projection(output, model, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="9")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sep-margin", type=float, default=SEP_MARGIN)
    args = parser.parse_args()
    for layer in [int(v) for v in args.layers.split(",")]:
        path = args.output / f"layer_{layer}.pt"
        if path.exists():
            raise ValueError(f"{path} 已存在")
    for layer in [int(v) for v in args.layers.split(",")]:
        meta = train(args.features, args.output / f"layer_{layer}.pt", layer, args.epochs,
                     args.batch_size, args.learning_rate, args.device, args.seed,
                     sep_margin=args.sep_margin)
        print(f"layer {layer}: epochs_run={meta['epochs_run']} best_epoch={meta['best_epoch']} "
              f"final_loss={meta['history'][-1]['loss']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
