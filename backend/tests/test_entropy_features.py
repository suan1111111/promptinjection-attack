from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.entropy_features import (EntropyFeatureStore, compute_scale, filter_candidates,
                                       shannon_entropy, window_sizes)
from services.gradient_features import GradientFeatureStore


class EntropyTests(unittest.TestCase):
    def test_entropy_known_distributions(self):
        self.assertEqual(shannon_entropy(np.zeros((9, 9), dtype=np.uint8)), 0)
        self.assertEqual(shannon_entropy(np.array([[0, 255]], dtype=np.uint8)), 1)
        self.assertEqual(shannon_entropy(np.arange(256, dtype=np.uint8)), 8)

    def test_windows_and_matlab_overwrite_crop(self):
        self.assertEqual(window_sizes((224, 224)), (8, 12, 16))
        self.assertEqual(window_sizes((1000, 2000)), (20, 30, 40))
        gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
        summary, details = compute_scale(gray, 8, 0)
        padded = np.pad(gray, 4, mode='symmetric')
        # Translate MATLAB one-based inclusive assignment explicitly.
        assignments, entropies = {}, []
        for y in range(1, padded.shape[0] - 8 + 1, 4):
            for x in range(1, padded.shape[1] - 8 + 1, 4):
                counts = np.unique(padded[y - 1:y + 8, x - 1:x + 8], return_counts=True)[1]
                probabilities = counts / counts.sum()
                entropy = -(probabilities * np.log2(probabilities)).sum()
                entropies.append(entropy)
                for yy in range(y, y + 9):
                    for xx in range(x, x + 9):
                        assignments[yy, xx] = entropy
        h = max(gray.shape[0], max(y for y, x in assignments))
        w = max(gray.shape[1], max(x for y, x in assignments))
        cy = int(np.floor((h - 5) / 2 + .5))
        cx = int(np.floor((w - 7) / 2 + .5))
        expected = np.array([[assignments.get((y, x), 0) for x in range(cx, cx + 7)] for y in range(cy, cy + 5)])
        np.testing.assert_allclose(details['heatmap'], expected, atol=1e-14)
        self.assertAlmostEqual(summary['entropy_mean'], np.mean(entropies))
        self.assertEqual(details['window_entropies'].shape, (2, 2))

    def test_constant_small_shapes(self):
        for shape in [(1, 1), (1, 30), (30, 1), (3, 4)]:
            for scale, size in enumerate(window_sizes(shape)):
                summary, details = compute_scale(np.zeros(shape, dtype=np.uint8), size, scale)
                self.assertEqual(summary['entropy_max'], 0)
                self.assertEqual(details['mask'].shape, shape)
                self.assertFalse(details['mask'].any())

    def test_component_filter_preserves_upstream_absolute_comparison(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True
        summary, cleaned = filter_candidates(mask)
        self.assertEqual(summary['candidate_region_count'], 0)
        mask[10:20, 10:20] = True
        summary, cleaned = filter_candidates(mask)
        self.assertEqual(summary['candidate_region_count'], 1)
        self.assertEqual(summary['largest_candidate_area_ratio'], .01)
        self.assertFalse(cleaned[0, 0])

    def test_partial_restart_cache_and_other_features(self):
        root = Path(__file__).resolve().parents[2] / 'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            image = Path(directory) / 'sample.png'
            Image.fromarray(np.random.default_rng(42).integers(0, 256, (25, 30, 3), dtype=np.uint8)).save(image)
            gradient = GradientFeatureStore(directory).extract(image)
            def interrupt(gray, size, scale):
                if scale == 1:
                    raise RuntimeError('interrupted second scale')
                return compute_scale(gray, size, scale)
            with patch('services.entropy_features.compute_scale', side_effect=interrupt):
                with self.assertRaises(RuntimeError):
                    EntropyFeatureStore(directory).extract(image)
            with patch('services.entropy_features.compute_scale', wraps=compute_scale) as computation:
                result = EntropyFeatureStore(directory).extract(image)
                self.assertEqual(computation.call_count, 2)
            with patch('services.entropy_features.compute_scale', side_effect=AssertionError('must reuse')):
                self.assertEqual(EntropyFeatureStore(directory).extract(image), result)
            self.assertEqual(GradientFeatureStore(directory).extract(image), gradient)
            self.assertEqual(len(result), 18)
