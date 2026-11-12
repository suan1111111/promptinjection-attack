"""CPU tests for the KCD detector on synthetic projected features."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcs_detector import KCDDetector, normalize


def make_split(rng, n_benign, n_malicious, separation=2.0):
    benign = rng.normal(0.0, 0.25, size=(n_benign, 256))
    malicious = rng.normal(separation, 0.25, size=(n_malicious, 256))
    features = np.vstack([benign, malicious]).astype(np.float32)
    labels = np.array([0] * n_benign + [1] * n_malicious)
    return features, labels


class KCDTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train_x, self.train_y = make_split(rng, 40, 40)
        self.val_x, self.val_y = make_split(rng, 10, 10)
        self.test_x, self.test_y = make_split(rng, 10, 10)

    def test_scores_separate_and_calibrate(self):
        detector = KCDDetector(k=5).fit(self.train_x, self.train_y)
        scores = detector.score(self.val_x)
        self.assertLess(scores[self.val_y == 0].mean(), 0)
        self.assertGreater(scores[self.val_y == 1].mean(), 0)
        calibration = detector.calibrate(self.val_x, self.val_y)
        self.assertGreater(calibration["combined"], 0.95)
        metrics = detector.evaluate(self.test_x, self.test_y)
        self.assertGreater(metrics["auroc"], 0.99)
        self.assertGreater(metrics["f1"], 0.9)
        self.assertGreaterEqual(metrics["threshold"], 0.0)

    def test_normalize_handles_zero_rows(self):
        features = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        out = normalize(features)
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), [0.0, 1.0]))

    def test_single_class_rejected(self):
        detector = KCDDetector(k=3)
        with self.assertRaises(ValueError):
            detector.fit(self.train_x, np.zeros(len(self.train_x), dtype=int))


if __name__ == "__main__":
    unittest.main()
