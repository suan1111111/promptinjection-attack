import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from app import create_app
from services.detection_ocr import recognize_grounding
from services.detection_tree import demo_tree


class DetectionOCRTests(unittest.TestCase):
    def setUp(self):
        self.raw = '<|ref|>text<|/ref|><|det|>[[100,100,500,500]]<|/det|>hello'
        stream = io.BytesIO()
        Image.new('RGB', (64, 64), 'white').save(stream, 'PNG')
        self.content = stream.getvalue()

    def test_adapter_input_and_cleanup_on_success_and_failure(self):
        paths = []
        def infer(**kwargs):
            path = Path(kwargs['image_path'])
            paths.append(path.parent)
            with Image.open(path) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.mode, 'RGB')
            self.assertEqual(kwargs['function_type'], 'markdown')
            self.assertFalse(kwargs['save_results'])
            return self.raw
        with patch('services.ocr_service.process_image', side_effect=infer):
            self.assertEqual(recognize_grounding(Image.new('RGB', (64, 64))), self.raw)
        self.assertFalse(paths[0].exists())
        def fail(**kwargs):
            paths.append(Path(kwargs['image_path']).parent)
            raise RuntimeError('model failed')
        with patch('services.ocr_service.process_image', side_effect=fail):
            with self.assertRaises(RuntimeError):
                recognize_grounding(Image.new('RGB', (64, 64)))
        self.assertFalse(paths[-1].exists())

    def test_default_application_auto_ocr_reuse_and_manual_override(self):
        app = create_app(enable_whisper=False)
        client = app.test_client()
        with TemporaryDirectory() as directory:
            app.config['DETECTION_RUN_STORE'] = directory
            with patch('services.ocr_service.process_image', return_value=self.raw) as infer:
                result = client.post('/api/detection/tree-runs', data={
                    'file': (io.BytesIO(self.content), 'image.png'),
                    'tree': json.dumps(demo_tree()), 'format': 'tsv',
                })
                self.assertEqual(result.status_code, 201)
                self.assertEqual(result.json['status'], 'unmatched')
                self.assertEqual(len(result.json['results']), 7)
                self.assertEqual(infer.call_count, 1)
                result = client.post('/api/detection/compute', data={
                    'file': (io.BytesIO(self.content), 'image.png'),
                    'operation_id': 'text.geometry', 'ocr_input': self.raw,
                })
                self.assertEqual(result.status_code, 200)
                self.assertEqual(infer.call_count, 1)

    def test_disabled_ocr_and_model_failure(self):
        app = create_app(enable_ocr=False, enable_whisper=False)
        self.assertIsNone(app.config.get('DETECTION_OCR_PROVIDER'))
        app.config['DETECTION_OCR_PROVIDER'] = recognize_grounding
        with TemporaryDirectory() as directory:
            app.config['DETECTION_RUN_STORE'] = directory
            with patch('services.ocr_service.process_image', side_effect=RuntimeError('model failed')):
                result = app.test_client().post('/api/detection/tree-runs', data={
                    'file': (io.BytesIO(self.content), 'image.png'), 'tree': json.dumps(demo_tree()),
                })
                self.assertEqual(result.status_code, 201)
                self.assertEqual(result.json['status'], 'incomplete')
                self.assertNotIn('ocr.structured', result.json['results'])
                self.assertIn('image.gradient', result.json['results'])
