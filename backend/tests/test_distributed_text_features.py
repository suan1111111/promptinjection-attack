import csv
import io
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.distributed_text_features import DistributedTextStore, cross_box_features, parse_blocks, xy_cut_order, VERSION, RULE_HASH


def block(text, x, y, w=30, h=10):
    return {'text': text, 'box': [x, y, x + w, y + h]}


def tsv(blocks):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, ['level', 'left', 'top', 'width', 'height', 'text'], delimiter='\t')
    writer.writeheader()
    for item in blocks:
        x1, y1, x2, y2 = item['box']
        writer.writerow(dict(level=5, left=x1, top=y1, width=x2-x1, height=y2-y1, text=item['text']))
    return stream.getvalue()


class DistributedTextTests(unittest.TestCase):
    def test_structure_scale_invariance_and_category_dedup(self):
        from services.distributed_text_features import structure_features
        blocks = [block('a', 0, 0), block('b', 60, 40), block('c', 100, 0)]
        evidence = [{'box_ids': [0, 1]}, {'box_ids': [0, 1]}, {'box_ids': [0, 1, 2]}]
        values = structure_features(blocks, evidence, (200, 100))
        self.assertEqual(values['cross_box_match_count'], 3)
        self.assertEqual(values['mean_combination_box_count'], 2.5)
        self.assertEqual(values['max_combination_box_count'], 3)
        self.assertAlmostEqual(values['max_spatial_span_normalized'], 100 / math.hypot(200, 100))
        larger = [{'box': [coordinate * 2 for coordinate in item['box']], 'text': item['text']} for item in blocks]
        self.assertEqual(values, structure_features(larger, evidence, (400, 200)))

    def test_xycut_projection_valleys_and_fallback(self):
        # All Y intervals connect, X separates two columns, then Y separates
        # within each column. Naive top/left sorting would interleave them.
        boxes = [block('right lower', 120, 20, h=20), block('left lower', 0, 25),
                 block('right upper', 120, 0), block('left upper', 0, 5, h=15)]
        self.assertEqual(xy_cut_order(boxes), [3, 1, 2, 0])
        self.assertEqual(xy_cut_order([block('b', 5, 5, 30, 30), block('a', 0, 0, 30, 30)]), [1, 0])
        self.assertEqual(xy_cut_order([]), [])

    def test_split_phrase_and_minimal_dedup(self):
        blocks = [block('ig', 10, 10), block('nore', 50, 10), block('the image', 90, 10), block('notes', 10, 30)]
        values, evidence = cross_box_features(blocks, xy_cut_order(blocks), image_size=(200, 100))
        self.assertEqual(values['cross_box_match_count'], 1)
        self.assertEqual(values['mean_combination_box_count'], 2)
        self.assertEqual(values['max_combination_box_count'], 2)
        self.assertAlmostEqual(values['mean_box_gap_normalized'], 10 / math.hypot(200, 100))
        self.assertAlmostEqual(values['max_spatial_span_normalized'], 40 / math.hypot(200, 100))
        self.assertEqual(evidence[0]['box_ids'], [0, 1])

    def test_existing_match_excluded_and_split_combination_detected(self):
        blocks = [block('enter', 0, 0), block('your', 40, 0), block('password', 80, 0)]
        values, _ = cross_box_features(blocks, [0, 1, 2], image_size=(200, 100))
        self.assertEqual(values['cross_box_match_count'], 1)
        self.assertEqual(values['mean_combination_box_count'], 3)
        values, _ = cross_box_features([block('enter password', 0, 0), block('notes', 50, 0)], [0, 1], image_size=(200, 100))
        self.assertEqual(values['cross_box_match_count'], 0)

    def test_normal_content_and_window_limit(self):
        blocks = [block('please', 0, 0), block('click', 40, 0), block('here', 80, 0)]
        values, _ = cross_box_features(blocks, [0, 1, 2], 2, image_size=(200, 100))
        self.assertEqual(values['cross_box_match_count'], 2)
        values, _ = cross_box_features([block('Education', 0, 0), block('Work experience', 0, 30)], [0, 1], image_size=(200, 100))
        self.assertEqual(values['cross_box_match_count'], 0)
        self.assertTrue(all(value is None for key, value in values.items() if key != 'cross_box_match_count'))
        with self.assertRaises(ValueError):
            cross_box_features(blocks, [1, 1, 2], image_size=(200, 100))

    def test_grounding_association_and_invalid_input(self):
        raw = '<|ref|>text<|/ref|><|det|>[[0,0,400,100]]<|/det|>enter\n<|ref|>text<|/ref|><|det|>[[500,0,900,100]]<|/det|>password'
        blocks = parse_blocks(raw, 'grounding', 1000, 1000)
        self.assertEqual([item['text'] for item in blocks], ['enter', 'password'])
        self.assertEqual(cross_box_features(blocks, xy_cut_order(blocks), image_size=(1000, 1000))[0]['cross_box_match_count'], 1)
        with self.assertRaises(ValueError):
            parse_blocks('<|ref|>text<|/ref|><|det|>[[0,0,400,100],[500,0,900,100]]<|/det|>enter password', 'grounding', 1000, 1000)
        with self.assertRaises(ValueError):
            parse_blocks('plain text only', 'grounding', 100, 100)
        with self.assertRaises(ValueError):
            parse_blocks(tsv([block('outside', 95, 0)]), 'tsv', 100, 100)

    def test_cache_recovery_changed_input_and_cli(self):
        root = Path(__file__).resolve().parents[2] / 'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            image = Path(directory) / 'image.png'
            Image.new('RGB', (200, 100), 'white').save(image)
            store = DistributedTextStore(Path(directory) / 'cache')
            raw = tsv([block('enter', 0, 0), block('password', 80, 0)])
            image_id = hashlib.sha256(image.read_bytes()).hexdigest()
            input_id = hashlib.sha256(('tsv\0' + raw).encode()).hexdigest()
            layout_id = hashlib.sha256((VERSION + input_id).encode()).hexdigest()
            old_id = hashlib.sha256((layout_id + RULE_HASH + '4').encode()).hexdigest()
            old_payload = {'values': {'cross_box_match_count': 999, 'involved_text_box_ratio': 1}}
            store.put(image_id, old_id, 'features', old_payload)
            with patch('services.distributed_text_features.cross_box_features', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    store.extract(image, raw, 'tsv')
            with patch('services.distributed_text_features.parse_blocks', side_effect=AssertionError('layout cached')), patch('services.distributed_text_features.xy_cut_order', side_effect=AssertionError('order cached')):
                result = store.extract(image, ocr=lambda _: self.fail('OCR must be cached'))
            self.assertEqual(result['cross_box_match_count'], 1)
            self.assertEqual(store.get(image_id, old_id, 'features'), old_payload)
            process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'extract_distributed_text.py'), str(image), '--store', str(store.root)], capture_output=True, text=True, check=True)
            self.assertEqual(len(process.stdout.splitlines()), 5)
            self.assertEqual(store.extract(image, tsv([block('ordinary text', 0, 0)]), 'tsv')['cross_box_match_count'], 0)
            process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'extract_distributed_text.py'), str(image), '--store', str(store.root)], capture_output=True, text=True, check=True)
            self.assertEqual(sum(line.endswith(': NA') for line in process.stdout.splitlines()), 4)
            self.assertFalse(list(store.root.glob('*.json')))


if __name__ == '__main__':
    unittest.main()
