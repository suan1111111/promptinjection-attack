from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.residual_features import CONFIG_ID, ResidualFeatureStore, compute_residual, summarize


class ResidualTests(unittest.TestCase):
    def test_constant_and_linear_ramp(self):
        y, x = np.indices((9, 11))
        for kind in ('kb', 'kv'):
            np.testing.assert_array_equal(compute_residual(np.full((9, 11), 180, np.uint8), kind), 0)
            np.testing.assert_array_equal(compute_residual(7 * x + 3 * y + 20, kind), 0)

    def test_author_residual_formulas(self):
        gray = np.random.default_rng(32).integers(0, 256, (9, 11)).astype(float)
        for kind, border in (('kb', 1), ('kv', 2)):
            expected = []
            for y in range(border, gray.shape[0] - border):
                row = []
                for x in range(border, gray.shape[1] - border):
                    at = lambda dy, dx: gray[y + dy, x + dx]
                    if kind == 'kb':
                        value = -4 * at(0, 0) + 2 * sum(at(dy, dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))) - sum(at(dy, dx) for dy in (-1, 1) for dx in (-1, 1))
                    else:
                        value = -12 * at(0, 0) + 8 * sum(at(dy, dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                        value -= 6 * sum(at(dy, dx) for dy in (-1, 1) for dx in (-1, 1))
                        value -= 2 * sum(at(dy, dx) for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)))
                        value += 2 * sum(at(dy, dx) for dy in (-2, -1, 1, 2) for dx in (-2, -1, 1, 2) if abs(dy) + abs(dx) == 3)
                        value -= sum(at(dy, dx) for dy in (-2, 2) for dx in (-2, 2))
                    row.append(value)
                expected.append(row)
            np.testing.assert_array_equal(compute_residual(gray, kind), expected)

    def test_summary_known_values(self):
        values = summarize(np.array([-2., 0., 2.]), 'kb')
        self.assertAlmostEqual(values['kb_residual_mean_abs'], 4 / 3)
        self.assertAlmostEqual(values['kb_residual_energy'], 8 / 3)
        self.assertAlmostEqual(values['kb_residual_std'], np.sqrt(8 / 3))

    def test_partial_failure_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.fromarray(np.random.default_rng(4).integers(0, 256, (20, 25), dtype=np.uint8)).save(path)
            store = ResidualFeatureStore(Path(directory) / 'cache')
            def interrupted(gray, kind):
                if kind == 'kv':
                    raise RuntimeError('interrupted')
                return compute_residual(gray, kind)
            with patch('services.residual_features.compute_residual', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    store.extract(path)
            image_id = hashlib.sha256(path.read_bytes()).hexdigest()
            with store._connect() as db:
                statuses = {name: json.loads(payload)['status'] for name, payload in db.execute('SELECT group_name,payload FROM features WHERE image_id=? AND config_id=?', (image_id, CONFIG_ID))}
            self.assertEqual(statuses, {'residual_kb': 'completed', 'residual_kv': 'failed'})
            with patch('services.residual_features.compute_residual', wraps=compute_residual) as compute:
                values = store.extract(path)
                self.assertEqual(compute.call_count, 1)
                self.assertEqual(compute.call_args.args[1], 'kv')
            with patch('services.residual_features.compute_residual', side_effect=AssertionError('recomputed')):
                self.assertEqual(store.extract(path), values)

    def test_small_image_preserves_kb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.png'
            Image.new('L', (3, 3), 200).save(path)
            store = ResidualFeatureStore(Path(directory) / 'cache')
            with self.assertRaises(ValueError):
                store.extract(path)
            self.assertEqual(store.extract(path, ('kb',))['kb_residual_energy'], 0)

    def test_transparent_pixels_use_white_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transparent.png'
            rgba = np.random.default_rng(3).integers(0, 256, (9, 9, 4), dtype=np.uint8)
            rgba[:, :, 3] = 0
            Image.fromarray(rgba).save(path)
            values = ResidualFeatureStore(Path(directory) / 'cache').extract(path)
            self.assertTrue(all(value == 0 for value in values.values()))


if __name__ == '__main__':
    unittest.main()
