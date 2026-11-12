"""Package the trained RCS detector into a reusable artifact.

Output:
  data/outputs/rcs_artifact_v2/
    projection_layer9.pt
    kcd_state.npz      benign/malicious normalised vectors, k, threshold
    meta.json          layer, feature config, metrics, model path
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from rcs_detector import KCDDetector, load_layer  # noqa: E402
from rcs_features import load_feature  # noqa: E402
from rcs_projection import load_projection, project_vectors  # noqa: E402

ROOT = Path(r"D:\CodexProjects\PrivaSee")
FEATURES = ROOT / "data/outputs/rcs_features_v2"
PROJECTIONS = ROOT / "data/outputs/rcs_projection_v2_m1_all"
OWN = ROOT / "data/outputs/rcs_features_100pairs"
OUT = ROOT / "data/outputs/rcs_artifact_v2"
LAYER = 9
K = 15

OUT.mkdir(parents=True, exist_ok=True)
model, projection_meta = load_projection(PROJECTIONS / f"layer_{LAYER}.pt")
split = load_layer(FEATURES, LAYER)

train_z = project_vectors(model, split["train"]["x"])
val_z = project_vectors(model, split["val"]["x"])
detector = KCDDetector(k=K).fit(train_z, split["train"]["y"])
calibration = detector.calibrate(val_z, split["val"]["y"])
val_scores = detector.score(val_z)
val_metrics = {
    "auroc": float(roc_auc_score(split["val"]["y"], val_scores)),
    "f1": float(f1_score(split["val"]["y"], (val_scores > detector.threshold).astype(int), zero_division=0)),
}

own = load_layer(OWN, LAYER)
own_x, own_y_list = [], []
own_config = json.loads((OWN / "config.json").read_text(encoding="utf-8"))
own_index = json.loads((OWN / "index.json").read_text(encoding="utf-8"))
own_pos = own_config["layers"].index(LAYER)
for record in own_index["records"]:
    if (record["status"] != "completed" or record["row"].get("domain") != "pixel"
            or record["row"].get("split") != "test"):
        continue
    vectors, _ = load_feature(OWN / record["file"])
    own_x.append(vectors[own_pos])
    own_y_list.append(int(record["row"]["label"]))
own_z = project_vectors(model, np.stack(own_x))
own_y = np.array(own_y_list)
own_scores = detector.score(own_z)
own_metrics = {
    "auroc": float(roc_auc_score(own_y, own_scores)),
    "f1": float(f1_score(own_y, (own_scores > detector.threshold).astype(int), zero_division=0)),
    "fpr": float(((own_scores > detector.threshold) & (own_y == 0)).sum() / max((own_y == 0).sum(), 1)),
}
safe_threshold = -0.05
safe_pred = own_scores > safe_threshold
recall_first = {
    "threshold_safe": safe_threshold,
    "recall": float((safe_pred & (own_y == 1)).sum() / max((own_y == 1).sum(), 1)),
    "false_positive_rate": float((safe_pred & (own_y == 0)).sum() / max((own_y == 0).sum(), 1)),
    "attacks": int((own_y == 1).sum()),
    "cleans": int((own_y == 0).sum()),
}

shutil.copy2(PROJECTIONS / f"layer_{LAYER}.pt", OUT / f"projection_layer{LAYER}.pt")
np.savez_compressed(
    OUT / "kcd_state.npz",
    benign=detector.benign.astype(np.float32),
    malicious=detector.malicious.astype(np.float32),
    threshold=np.float32(detector.threshold),
    k=np.int32(K),
    layer=np.int32(LAYER),
)

config = json.loads((FEATURES / "config.json").read_text(encoding="utf-8"))
meta = {
    "schema": 1,
    "layer": LAYER,
    "k": K,
    "threshold": float(detector.threshold),
    "threshold_safe": -0.05,
    "threshold_safe_note": "recall-first filter: below this the image is dropped as safe; above it goes to the next layer",
    "feature_config_fingerprint": config["fingerprint"],
    "model_dir": str(ROOT / "models/qwen2.5-vl-3b-instruct"),
    "quantization": "NF4 double quant, BF16 compute",
    "min_visual_tokens": config["min_visual_tokens"],
    "max_visual_tokens": config["max_visual_tokens"],
    "projection": f"projection_layer{LAYER}.pt",
    "projection_meta": {
        "best_epoch": projection_meta["best_epoch"],
        "best_val_auroc": projection_meta["best_val_auroc"],
        "sep_margin": projection_meta.get("sep_margin"),
        "sep_weight": projection_meta.get("sep_weight"),
        "dataset_weight": projection_meta.get("dataset_weight"),
        "variance_weight": projection_meta.get("variance_weight"),
    },
    "calibration": calibration,
    "val_metrics": val_metrics,
    "own_pixel_test_metrics": own_metrics,
    "own_pixel_test_recall_first": recall_first,
    "question_hint": "use the same question as the feature-extraction stage; text pairs used 'Describe this image.'",
}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"val": val_metrics, "own_pixel_test": own_metrics, "threshold": meta["threshold"]}, indent=1))
