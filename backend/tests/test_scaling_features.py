import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from services.scaling_features import ScalingFeatureStore, peak_spectrum, down_up_psnr


class ScalingTests(unittest.TestCase):
    def test_nearest_hidden_payload_and_scores(self):
        clean = np.full((256, 256, 3), 128, dtype=np.uint8)
        payload = np.repeat(((np.indices((64, 64)).sum(axis=0) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)
        attack = clean.copy()
        attack[2::4, 2::4] = payload
        np.testing.assert_array_equal(np.asarray(Image.fromarray(attack).resize((64, 64), Image.Resampling.NEAREST)), payload)
        self.assertGreater(peak_spectrum(attack, (64, 64), 0), peak_spectrum(clean, (64, 64), 0))
        self.assertTrue(np.isinf(down_up_psnr(clean, (64, 64), 'nearest')))
        self.assertLess(down_up_psnr(attack, (64, 64), 'nearest'), 20)

    def test_known_psnr(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[1, 1] = 255
        self.assertAlmostEqual(down_up_psnr(rgb, (1, 1), 'nearest'), 10 * np.log10(4 / 3))

    def test_no_predicted_peaks(self):
        self.assertIsNone(peak_spectrum(np.zeros((256, 256, 3), dtype=np.uint8), (224, 224)))

    def test_partial_cache_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8)).save(image)
            store = ScalingFeatureStore(Path(directory) / 'cache')
            with patch('services.scaling_features.down_up_psnr', side_effect=RuntimeError('interrupted')):
                first, failures = store.extract(image, (64, 64))
            self.assertIn('down_up_psnr_db', failures)
            with patch('services.scaling_features.peak_spectrum', side_effect=AssertionError('should use cache')):
                second, failures = store.extract(image, (64, 64))
            self.assertFalse(failures)
            self.assertEqual(first['peak_spectrum_percentile'], second['peak_spectrum_percentile'])
            self.assertTrue(np.isinf(second['down_up_psnr_db']))
            with patch('services.scaling_features.peak_spectrum', return_value=123) as compute:
                third, failures = store.extract(image, (64, 64), radius=0)
            compute.assert_called_once()
            self.assertEqual(third['peak_spectrum_percentile'], 123)
            with self.assertRaises(ValueError):
                store.extract(image, (512, 512))


if __name__ == '__main__':
    unittest.main()
