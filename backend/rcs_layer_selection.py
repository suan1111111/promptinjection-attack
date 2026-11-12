"""Select safety-critical layers from cached RCS hidden states.

Input is the feature directory produced by ``rcs_features.py`` (one .npz per
row, [layers, 2048] float32). For every layer we compute geometric separation
metrics, normalise them across layers, and rank the layers. Nothing here loads
a VLM; it runs on CPU over cached vectors.

Metrics (all on StandardScaler-normalised features):
  svm_margin      linear SVM margin 2/||w||
  silhouette      mean silhouette coefficient
  distance_ratio  between-centroid distance / mean within-class distance
  paired_gap      mean ||z_attack - z_clean|| over pairs, divided by the
                  mean pairwise distance of all samples (scale free)
  mmd             RBF-kernel MMD, reported as a reference, not ranked
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.spatial.distance import pdist

from rcs_features import digest, load_feature, output_on_d

PRIMARY = ("svm_margin", "silhouette", "distance_ratio", "paired_gap")
REFERENCE = ("mmd",)


def collect(feature_dir: Path, splits: set[str] | None = None):
    feature_dir = Path(feature_dir).resolve()
    config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((feature_dir / "index.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in config.items() if k != "fingerprint"}) != config["fingerprint"]:
        raise ValueError("缓存配置指纹校验失败")
    if index["config_fingerprint"] != config["fingerprint"]:
        raise ValueError("index 与 config 指纹不一致")
    layers = list(config["layers"])
    matrices: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    labels: list[int] = []
    groups: list[str] = []
    pairs: list[str] = []
    for record in index["records"]:
        if record["status"] != "completed":
            continue
        row = record["row"]
        if splits and row.get("split") not in splits:
            continue
        vectors, meta = load_feature(feature_dir / record["file"])
        if meta["layers"] != layers:
            raise ValueError("特征层号与 config 不一致")
        for layer in layers:
            matrices[layer].append(vectors[layers.index(layer)])
        labels.append(int(row["label"]))
        groups.append(row.get("group") or row["id"])
        pairs.append(row.get("pair_id") or row["id"])
    if len(set(labels)) < 2:
        raise ValueError("选层需要同时包含良性和恶意样本")
    return (
        layers,
        {layer: np.stack(values) for layer, values in matrices.items()},
        np.array(labels),
        groups,
        pairs,
        config,
    )


def _scaled(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def svm_margin(X: np.ndarray, y: np.ndarray) -> float:
    model = SVC(kernel="linear", C=1.0).fit(X, y)
    norm = float(np.linalg.norm(model.coef_))
    return 2.0 / norm if norm > 0 else 0.0


def silhouette(X: np.ndarray, y: np.ndarray) -> float:
    return float(silhouette_score(X, y))


def distance_ratio(X: np.ndarray, y: np.ndarray) -> float:
    a, b = X[y == 0], X[y == 1]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    intra_a = float(pdist(a).mean()) if len(a) > 1 else 0.0
    intra_b = float(pdist(b).mean()) if len(b) > 1 else 0.0
    intra = (intra_a + intra_b) / 2
    return inter / intra if intra else 0.0


def paired_gap(X: np.ndarray, y: np.ndarray, pairs: list[str]) -> float:
    by_pair: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for index, pair in enumerate(pairs):
        by_pair[pair][int(y[index])] = X[index]
    distances = [
        float(np.linalg.norm(v[1] - v[0]))
        for v in by_pair.values()
        if 0 in v and 1 in v
    ]
    if not distances:
        return 0.0
    baseline = float(pdist(X).mean()) if len(X) > 1 else 0.0
    return float(np.mean(distances) / baseline) if baseline else 0.0


def mmd(X: np.ndarray, y: np.ndarray) -> float:
    a, b = X[y == 0], X[y == 1]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    gamma = 1.0 / X.shape[1]
    kaa, kbb, kab = rbf_kernel(a, a, gamma=gamma), rbf_kernel(b, b, gamma=gamma), rbf_kernel(a, b, gamma=gamma)
    term_a = (kaa.sum() - np.trace(kaa)) / (len(a) * (len(a) - 1))
    term_b = (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1))
    return float(np.sqrt(max(0.0, term_a + term_b - 2 * kab.mean())))


def layer_metrics(X: np.ndarray, y: np.ndarray, pairs: list[str]) -> dict[str, float]:
    Z = _scaled(X)
    return {
        "svm_margin": svm_margin(Z, y),
        "silhouette": silhouette(Z, y),
        "distance_ratio": distance_ratio(Z, y),
        "paired_gap": paired_gap(Z, y, pairs),
        "mmd": mmd(Z, y),
    }


BOOTSTRAP_METRICS = ("distance_ratio", "paired_gap")


def bootstrap_layer_metrics(X: np.ndarray, y: np.ndarray, pairs: list[str]) -> dict[str, float]:
    """Cheap metric subset used inside the group bootstrap (no SVM fits)."""
    Z = _scaled(X)
    return {"distance_ratio": distance_ratio(Z, y), "paired_gap": paired_gap(Z, y, pairs)}


def ranks(values: dict[int, float]) -> dict[int, int]:
    """Rank 1 = best (largest value)."""
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return {layer: position for position, (layer, _) in enumerate(ordered, 1)}


def composite(metrics: dict[int, dict[str, float]], names=PRIMARY) -> dict[int, float]:
    per_metric = {name: ranks({layer: metrics[layer][name] for layer in metrics}) for name in names}
    return {layer: float(np.mean([per_metric[name][layer] for name in names])) for layer in metrics}


def select(feature_dir: Path, output: Path, top_k: int = 3, splits: set[str] | None = None,
           bootstrap: int = 100, seed: int = 42) -> dict:
    layers, matrices, y, groups, pairs, config = collect(feature_dir, splits)
    metrics = {layer: layer_metrics(matrices[layer], y, pairs) for layer in layers}
    score = composite(metrics)
    ranking = sorted(layers, key=lambda layer: score[layer])
    top = ranking[:top_k]

    frequencies = {layer: 0 for layer in layers}
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        unique_groups = sorted(set(groups))
        group_index = {group: [i for i, g in enumerate(groups) if g == group] for group in unique_groups}
        candidates = ranking[: max(top_k * 3, 10)]
        for _ in range(bootstrap):
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate([group_index[g] for g in chosen])
            if len(set(y[indices])) < 2:
                continue
            boot_pairs = [pairs[i] for i in indices]
            boot_metrics = {layer: bootstrap_layer_metrics(matrices[layer][indices], y[indices], boot_pairs)
                            for layer in candidates}
            boot_score = composite(boot_metrics, names=BOOTSTRAP_METRICS)
            for layer in sorted(candidates, key=lambda l: boot_score[l])[:top_k]:
                frequencies[layer] += 1

    output = output_on_d(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in ranking:
        rows.append({
            "layer": layer,
            **{name: round(metrics[layer][name], 6) for name in PRIMARY + REFERENCE},
            "composite_rank": round(score[layer], 3),
            "rank": ranking.index(layer) + 1,
            "topk_frequency": frequencies[layer],
        })
    (output / "layer_scores.json").write_text(json.dumps({
        "schema": 1,
        "feature_config": config["fingerprint"],
        "samples": int(len(y)),
        "groups": len(set(groups)),
        "splits": sorted(splits) if splits else "all",
        "primary_metrics": list(PRIMARY),
        "bootstrap_metrics": list(BOOTSTRAP_METRICS),
        "bonferroni_note": "top-k frequency = fraction of group-bootstrap resamples where the layer stayed in top-k",
        "bootstrap": bootstrap,
        "top_k": top,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    with (output / "layer_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = __import__("csv").writer(handle)
        writer.writerow(["rank", "layer", *PRIMARY, *REFERENCE, "composite_rank", "topk_frequency"])
        for row in rows:
            writer.writerow([row["rank"], row["layer"], *[row[name] for name in PRIMARY + REFERENCE],
                             row["composite_rank"], row["topk_frequency"]])
    return {"top_k": top, "rows": rows, "samples": len(y), "groups": len(set(groups))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--splits", default="all", help="comma separated, e.g. train,val")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits = None if args.splits == "all" else {value.strip() for value in args.splits.split(",")}
    result = select(args.features, args.output, args.top_k, splits, args.bootstrap, args.seed)
    print("samples:", result["samples"], "groups:", result["groups"])
    for row in result["rows"][: args.top_k]:
        print(f"  layer {row['layer']:>2}  rank={row['rank']}  composite={row['composite_rank']}  "
              f"topk_freq={row['topk_frequency']}/{args.bootstrap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
