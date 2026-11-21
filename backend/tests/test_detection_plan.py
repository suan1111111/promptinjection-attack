import io
import json
import unittest
from unittest.mock import patch
from PIL import Image
from flask import Flask
from services.detection_plan import execute_order, validate_order
from services.detection_api import detection_bp


class PlanTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new('RGB', (20, 20), 'white').save(buffer, format='PNG')
        self.content = buffer.getvalue()
        self.raw = '<|ref|>text<|/ref|><|det|>[[0,0,500,500]]<|/det|>hello'
        self.order = ['ocr.structured', 'text.rules', 'image.gray', 'image.gradient', 'text.gradient_overlap']

    def test_fixed_order_direct_reuse(self):
        calls = []
        def provider(image):
            calls.append(1)
            return self.raw
        with patch('services.detection_compute.ComputationSession.compute', side_effect=AssertionError('recursive call')):
            result = execute_order(self.content, self.order + ['ocr.structured'], ocr_provider=provider)
        self.assertEqual(calls, [1])
        self.assertEqual([r['operation_id'] for r in result['computations']], self.order + ['ocr.structured'])
        self.assertTrue(result['computations'][-1]['reused'])
        self.assertEqual(result['computations'][4]['reads'], ['ocr.structured', 'image.gradient'])

    def test_validate_before_computing(self):
        with patch('services.detection_plan.ComputationSession', side_effect=AssertionError('started')):
            with self.assertRaises(ValueError):
                execute_order(self.content, ['ocr.structured', 'image.gradient'])
        for value in (None, [], 'image.gray', [None], ['unknown'], ['text.rules', 'ocr.structured']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_order(value)

    def test_failure_blocks_dependents_but_not_independent_branch(self):
        with self.assertLogs('services.detection_plan', level='ERROR'):
            result = execute_order(self.content, self.order)
        self.assertEqual([r['status'] for r in result['computations']],
                         ['failed', 'blocked', 'completed', 'completed', 'blocked'])
        self.assertNotIn('text.gradient_overlap', result['results'])
        self.assertEqual(result['status'], 'incomplete')

    def test_sessions_are_isolated(self):
        execute_order(self.content, ['ocr.structured'], self.raw)
        with self.assertLogs('services.detection_plan', level='ERROR'):
            result = execute_order(self.content, ['ocr.structured'])
        self.assertEqual(result['results'], {})

    def test_endpoints(self):
        app = Flask(__name__)
        app.register_blueprint(detection_bp)
        client = app.test_client()
        self.assertEqual(client.post('/api/detection/plans/validate', json={'operations': self.order}).status_code, 200)
        self.assertEqual(client.post('/api/detection/plans/validate', json={'operations': ['text.rules']}).status_code, 400)
        response = client.post('/api/detection/compute-sequence', data={
            'file': (io.BytesIO(self.content), 'image.png'), 'operations': json.dumps(self.order), 'ocr_input': self.raw})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'completed')
        self.assertEqual(len(response.json['results']), 5)
