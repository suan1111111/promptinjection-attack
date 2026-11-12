"""Tests for the packaged RCS detector artifact."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_features import PROJECT_ROOT
from rcs_projection import save_projection
from rcs_projection_core import LearnedProjection
from rcs_infer import RCSDetector


class InferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rcs_infer_", dir=PROJECT_ROOT / "tmp")
        self.dir = Path(self.temp.name)
        rng = np.random.default_rng(0)
        axis = np.zeros(256, dtype=np.float32)
        axis[0] = 1.0
        benign = axis + rng.normal(scale=0.01, size=(20, 256)).astype(np.float32)
        malicious = -axis + rng.normal(scale=0.01, size=(20, 256)).astype(np.float32)
        benign /= np.linalg.norm(benign, axis=1, keepdims=True)
        malicious /= np.linalg.norm(malicious, axis=1, keepdims=True)
        np.savez_compressed(self.dir / "kcd_state.npz", benign=benign, malicious=malicious,
                            threshold=np.float32(0.0), k=np.int32(5), layer=np.int32(9))
        model = LearnedProjection(input_dim=2048)
        save_projection(self.dir / "projection_layer9.pt", model,
                        {"layer": 9, "architecture": {"input_dim": 2048, "output_dim": 256, "hidden_dim": 512, "dropout": 0.3}})
        (self.dir / "meta.json").write_text(json.dumps({
            "projection": "projection_layer9.pt", "min_visual_tokens": 256, "max_visual_tokens": 768,
        }), encoding="utf-8")
        with np.load(self.dir / "kcd_state.npz") as state:
            self.benign = state["benign"].copy()
            self.malicious = state["malicious"].copy()

    def tearDown(self):
        self.temp.cleanup()

    def test_scores_have_expected_sign(self):
        detector = RCSDetector(self.dir)
        self.assertLess(detector.score_projected(self.benign[0]), 0.0)
        self.assertGreater(detector.score_projected(self.malicious[0]), 0.0)

    def test_label_threshold(self):
        detector = RCSDetector(self.dir)
        result = detector.score_image  # noqa: F841  (attribute exists)
        self.assertEqual(detector.layer, 9)
        self.assertEqual(detector.k, 5)

    def test_rejects_layer_mismatch(self):
        with np.load(self.dir / "kcd_state.npz") as state:
            arrays = {key: state[key] for key in state.files}
        arrays["layer"] = np.int32(18)
        np.savez_compressed(self.dir / "kcd_state.npz", **arrays)
        with self.assertRaises(ValueError):
            RCSDetector(self.dir)


if __name__ == "__main__":
    unittest.main()
