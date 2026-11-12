from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.ocr_numeric_features import OCRNumericStore, geometry, overlap, parse_grounding, parse_tsv, quality


class OCRNumericTests(unittest.TestCase):
    def test_grounding_and_missing_confidence(self):
        data = parse_grounding('<|ref|>text<|/ref|><|det|>[[0,0,999,999]]<|/det|>hello world', 20, 10)
        self.assertEqual(data['boxes'][0]['box'], [0, 0, 20, 10])
        self.assertEqual(quality(data)['character_count'], 10)
        self.assertIsNone(quality(data)['confidence_mean'])
        with self.assertRaises(ValueError):
            parse_grounding('plain text', 20, 10)

    def test_tsv_confidence_and_bounds(self):
        header = 'level\tleft\ttop\twidth\theight\tconf\ttext\n'
        data = parse_tsv(header + '5\t2\t2\t4\t3\t90\thello\n', 10, 10)
        self.assertEqual(quality(data)['confidence_mean'], .9)
        with self.assertRaises(ValueError):
            parse_tsv(header + '5\t9\t2\t4\t3\t90\thello\n', 10, 10)

    def test_union_geometry_and_contrast(self):
        data = {'boxes': [{'box': [2, 2, 6, 6]}, {'box': [4, 2, 8, 6]}]}
        gray = np.ones((10, 10))
        gray[3, 3] = 0
        values = geometry(data, gray)
        self.assertAlmostEqual(values['text_area_ratio'], .24)
        self.assertAlmostEqual(values['relative_box_height_mean'], .4)
        self.assertEqual(values['edge_text_box_ratio'], 0)
        self.assertEqual(values['box_background_contrast_mean'], .5)

    def test_overlap_denominators_and_empty(self):
        values = overlap(np.array([[True, True, False]]), np.array([[False, True, True]]), 'test')
        self.assertEqual(values['test_text_coverage'], .5)
        self.assertEqual(values['test_response_in_text_ratio'], .5)
        self.assertAlmostEqual(values['test_iou'], 1 / 3)
        self.assertTrue(all(v is None for v in overlap(np.zeros((1, 1), bool), np.zeros((1, 1), bool), 'test').values()))

    def test_partial_failure_resume_and_full_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (32, 32), 'white').save(path)
            store = OCRNumericStore(Path(directory) / 'cache')
            raw = '<|ref|>text<|/ref|><|det|>[[100,100,500,500]]<|/det|>hello'
            with patch.object(store, '_entropy', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    store.extract(path, raw=raw)
            with patch.object(store, '_gradient', side_effect=AssertionError('gradient reran')), patch('services.ocr_numeric_features.quality', side_effect=AssertionError('quality reran')):
                values = store.extract(path, ocr=lambda _: self.fail('OCR reran'))
            self.assertEqual(len(values), 16)
            with patch.object(store, '_entropy', side_effect=AssertionError('entropy reran')):
                self.assertEqual(store.extract(path), values)


if __name__ == '__main__':
    unittest.main()
