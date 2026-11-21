import io
import unittest
from unittest.mock import patch
from PIL import Image
from flask import Flask
from services.detection_compute import ComputationSession
from services.detection_api import detection_bp


class ComputeTests(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(stream, format='PNG')
        self.content = stream.getvalue()
        self.raw = '<|ref|>text<|/ref|><|det|>[[100,100,500,500]]<|/det|>hello'

    def test_geometry_does_not_compute_contrast_or_gradient(self):
        session = ComputationSession(self.content, self.raw)
        with patch('services.detection_compute.compute_gradient', side_effect=AssertionError), \
             patch('services.detection_compute.contrast_only', side_effect=AssertionError):
            self.assertGreater(session.compute('text.geometry')['text_area_ratio'], 0)
        self.assertEqual(list(session.results), ['ocr.structured', 'text.geometry'])

    def test_shared_dependencies_and_overlap(self):
        calls = []
        def provider(image):
            calls.append(image.size)
            return self.raw
        session = ComputationSession(self.content, ocr_provider=provider)
        session.compute('text.rules')
        result = session.compute('text.gradient_overlap')
        session.compute('text.gradient_overlap')
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['gradient_text_coverage'], 0)
        self.assertIsNone(result['gradient_response_in_text_ratio'])
        self.assertEqual(len(session.events), 5)
        self.assertEqual(session.results['ocr.structured']['boxes'][0]['text'], 'hello')

    def test_endpoint_and_missing_ocr(self):
        app = Flask(__name__)
        app.register_blueprint(detection_bp)
        client = app.test_client()
        def request(operation, **extra):
            return client.post('/api/detection/compute', data={
                'file': (io.BytesIO(self.content), 'image.png'), 'operation_id': operation, **extra})
        self.assertEqual(request('text.geometry').status_code, 422)
        self.assertEqual(request('missing').status_code, 400)
        result = request('text.geometry', ocr_input=self.raw)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json['computations']), 2)
        self.assertEqual(request('image.gradient').status_code, 200)


if __name__ == '__main__':
    unittest.main()
