"""CPU tests for layer selection on synthetic cached features."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_features import PROJECT_ROOT, digest, save_feature
from rcs_layer_selection import select

LAYERS = [0, 1, 2, 3, 4]
SIGNAL_LAYER = 3


def build_feature_dir(root: Path) -> Path:
    feature_dir = root / "features"
    rng = np.random.default_rng(0)
    records = []
    for index in range(10):
        pair = f"p{index}"
        group = f"g{index}"
        base = rng.normal(size=(len(LAYERS), 2048)).astype(np.float32)
        for label in (0, 1):
            vectors = base.copy()
            if label == 1:
                vectors[SIGNAL_LAYER, :8] += 5.0
            record_id = f"{pair}_{label}"
            row = {"id": record_id, "label": label, "group": group, "pair_id": pair, "split": "train"}
            meta = {"layers": LAYERS, "row": row, "signature": "sig", "config_fingerprint": "fp",
                    "image_sha256": "hash"}
            save_feature(feature_dir / "features" / f"{record_id}.npz", vectors, meta)
            records.append({"id": record_id, "row": row, "file": f"features/{record_id}.npz",
                            "status": "completed", "signature": "sig"})
    config = {"schema": 1, "layers": LAYERS, "dimension": 2048}
    config["fingerprint"] = digest(config)
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (feature_dir / "index.json").write_text(
        json.dumps({"schema": 1, "config_fingerprint": config["fingerprint"], "records": records}),
        encoding="utf-8")
    return feature_dir


class LayerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="layer_sel_", dir=PROJECT_ROOT / "tmp")
        self.root = Path(self.temp.name)
        self.feature_dir = build_feature_dir(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_signal_layer_is_selected(self):
        result = select(self.feature_dir, self.root / "out", top_k=3, bootstrap=0)
        self.assertIn(SIGNAL_LAYER, result["top_k"])
        best_paired = max(result["rows"], key=lambda row: row["paired_gap"])
        self.assertEqual(best_paired["layer"], SIGNAL_LAYER)
        payload = json.loads((self.root / "out" / "layer_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["samples"], 20)
        self.assertEqual(payload["groups"], 10)

    def test_bootstrap_is_deterministic(self):
        first = select(self.feature_dir, self.root / "out1", top_k=1, bootstrap=10, seed=7)
        second = select(self.feature_dir, self.root / "out2", top_k=1, bootstrap=10, seed=7)
        self.assertEqual(first["rows"], second["rows"])

    def test_bad_fingerprint_rejected(self):
        config = json.loads((self.feature_dir / "config.json").read_text(encoding="utf-8"))
        config["fingerprint"] = "wrong"
        (self.feature_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(ValueError):
            select(self.feature_dir, self.root / "out", bootstrap=0)


if __name__ == "__main__":
    unittest.main()
