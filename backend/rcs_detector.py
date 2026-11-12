"""KCD (K-nearest Contrastive Detection) on projected RCS features.

For one layer: load cached 2048-d features, project to 256-d, fit KCD on the
train split (benign vs malicious), calibrate a threshold on val, then report
val/test metrics. Layers are compared by validation AUROC.

KCD score (upstream eq. 8): after L2 normalisation,
    s(x) = ||z - z_(k)^benign|| - ||z - z_(k)^malicious||
Positive scores mean "closer to malicious". Higher is more suspicious.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from rcs_features import load_feature, output_on_d
from rcs_projection import load_projection, project_vectors


def load_layer(feature_dir: Path, layer: int) -> dict[str, dict]:
    feature_dir = Path(feature_dir).resolve()
    config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((feature_dir / "index.json").read_text(encoding="utf-8"))
    if index["config_fingerprint"] != config["fingerprint"]:
        raise ValueError("index 与 config 指纹不一致")
    if layer not in config["layers"]:
        raise ValueError(f"缓存里没有第 {layer} 层")
    position = config["layers"].index(layer)
    data = {"train": {"x": [], "y": []}, "val": {"x": [], "y": []}, "test": {"x": [], "y": []}}
    for record in index["records"]:
        if record["status"] != "completed":
            continue
        row = record["row"]
        split = row.get("split")
        if split not in data:
            continue
        vectors, _ = load_feature(feature_dir / record["file"])
        data[split]["x"].append(vectors[position])
        data[split]["y"].append(int(row["label"]))
    return {
        split: {"x": np.stack(values["x"]), "y": np.array(values["y"])}
        for split, values in data.items()
        if values["x"]
    }


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


class KCDDetector:
    def __init__(self, k: int = 15):
        if k < 1:
            raise ValueError("k 必须 >= 1")
        self.k = k
        self.benign: np.ndarray | None = None
        self.malicious: np.ndarray | None = None
        self.threshold = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "KCDDetector":
        if len(set(labels.tolist())) < 2:
            raise ValueError("KCD 训练需要同时包含良性(0)和恶意(1)")
        z = normalize(features)
        self.benign = z[labels == 0]
        self.malicious = z[labels == 1]
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.benign is None or self.malicious is None:
            raise RuntimeError("先调用 fit")
        z = normalize(features)
        k_benign = min(self.k, len(self.benign))
        k_malicious = min(self.k, len(self.malicious))
        d_benign = np.sort(cdist(z, self.benign), axis=1)[:, k_benign - 1]
        d_malicious = np.sort(cdist(z, self.malicious), axis=1)[:, k_malicious - 1]
        return d_benign - d_malicious

    def calibrate(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """Maximise 0.8 * balanced accuracy + 0.2 * F1 on the validation split."""
        scores = self.score(features)
        grid = np.quantile(scores, np.linspace(0.01, 0.99, 199))
        best = {"threshold": float(np.median(scores)), "combined": -1.0, "f1": 0.0, "balanced_accuracy": 0.0}
        for threshold in grid:
            predictions = (scores > threshold).astype(int)
            balanced = balanced_accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, zero_division=0)
            combined = 0.8 * balanced + 0.2 * f1
            if combined > best["combined"]:
                best = {"threshold": float(threshold), "combined": float(combined),
                        "f1": float(f1), "balanced_accuracy": float(balanced)}
        self.threshold = best["threshold"]
        return best

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict:
        scores = self.score(features)
        predictions = (scores > self.threshold).astype(int)
        result = {
            "samples": int(len(labels)),
            "benign": int((labels == 0).sum()),
            "malicious": int((labels == 1).sum()),
            "threshold": float(self.threshold),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        result["tpr"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        result["fpr"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
        if len(set(labels.tolist())) > 1:
            result["auroc"] = float(roc_auc_score(labels, scores))
            result["auprc"] = float(average_precision_score(labels, scores))
        else:
            result["auroc"] = None
            result["auprc"] = None
        return result


def run(feature_dir: Path, projections: Path, layers: list[int], output: Path, k: int = 15) -> dict:
    output = output_on_d(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"k": k, "layers": {}, "protocol": "fit on train, calibrate on val, report test"}
    for layer in layers:
        split = load_layer(feature_dir, layer)
        model, meta = load_projection(projections / f"layer_{layer}.pt")
        if meta["layer"] != layer:
            raise ValueError(f"投影层号不匹配：{meta['layer']} != {layer}")
        projected = {name: project_vectors(model, values["x"]) for name, values in split.items()}
        detector = KCDDetector(k=k).fit(projected["train"], split["train"]["y"])
        calibration = detector.calibrate(projected["val"], split["val"]["y"])
        val = detector.evaluate(projected["val"], split["val"]["y"])
        test = detector.evaluate(projected["test"], split["test"]["y"]) if "test" in projected else None
        record = {"layer": layer, "calibration": calibration, "val": val, "test": test,
                  "train_samples": int(len(split["train"]["y"]))}
        summary["layers"][layer] = record
        (output / f"layer_{layer}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"layer {layer}: val AUROC={val['auroc']:.4f} F1={val['f1']:.4f} "
              f"(thr={detector.threshold:.4f}, k={k})", flush=True)

    best_layer = max(summary["layers"], key=lambda layer: (
        summary["layers"][layer]["val"]["auroc"] or 0.0,
        summary["layers"][layer]["val"]["f1"],
    ))
    summary["best_layer"] = best_layer
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"best layer by val AUROC: {best_layer}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--layers", default="13,14,15")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",")]
    run(args.features, args.projections, layers, args.output, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
