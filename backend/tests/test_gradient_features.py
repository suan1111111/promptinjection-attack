import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.gradient_features import GradientFeatureStore, compute_gradient


class GradientTests(unittest.TestCase):
    def test_windows_and_union_match_brute_force(self):
        gray = np.random.default_rng(4).random((32, 37))
        summary, windows = compute_gradient(gray)
        gy, gx = np.gradient(gray)
        magnitude = np.hypot(gx, gy)
        normalized = (magnitude - magnitude.min()) / np.ptp(magnitude)
        expected = []
        coverage = np.zeros(gray.shape, dtype=bool)
        for y in windows["window_y"]:
            row = []
            for x in windows["window_x"]:
                mean = normalized[y:y + 15, x:x + 15].mean()
                row.append(mean)
                if mean > .1:
                    coverage[y:y + 15, x:x + 15] = True
            expected.append(row)
        np.testing.assert_allclose(windows["window_means"], expected, atol=1e-14)
        self.assertAlmostEqual(summary["high_gradient_area_ratio"], coverage.mean())
        self.assertAlmostEqual(summary["high_gradient_window_ratio"], (np.array(expected) > .1).mean())

    def test_constant_and_small_images(self):
        for shape in ((1, 1), (1, 22), (22, 1), (7, 8), (30, 30)):
            summary, _ = compute_gradient(np.ones(shape))
            self.assertEqual(list(summary.values()), [0., 0., 0.])

    def test_restart_cache_and_cli(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(np.random.default_rng(6).integers(0, 256, (35, 40, 3), dtype=np.uint8)).save(path)
            store = GradientFeatureStore(directory)
            first = store.extract(path)
            with patch("services.gradient_features.compute_gradient", side_effect=AssertionError("cached")):
                self.assertEqual(GradientFeatureStore(directory).extract(path), first)
            self.assertEqual(len(list(Path(directory).rglob("windows.npz"))), 1)
            with np.load(next(Path(directory).rglob("windows.npz"))) as windows:
                self.assertIn("window_means", windows)
            output = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "extract_gradient.py"),
                                     str(path), "--store", directory], capture_output=True, text=True, check=True)
            self.assertEqual(len(output.stdout.splitlines()), 3)
            for line in output.stdout.splitlines():
                name, value = line.split(": ")
                self.assertAlmostEqual(float(value), first[name], places=10)
            with self.assertRaisesRegex(ValueError, "not a directory"):
                store.extract(directory)


if __name__ == "__main__":
    unittest.main()
