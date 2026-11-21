import io
import json
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from flask import Flask
from services.detection_tree import demo_tree, execute_tree, validate_tree, evaluate
from services.detection_api import detection_bp


class TreeTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(buffer, format='PNG')
        self.content = buffer.getvalue()
        self.raw = '<|ref|>text<|/ref|><|det|>[[100,100,500,500]]<|/det|>hello'

    def test_no_match_completes_declared_features(self):
        report = execute_tree(self.content, demo_tree(), self.raw)
        self.assertEqual(report['status'], 'unmatched')
        self.assertEqual(len(report['results']), 7)
        self.assertTrue(report['completion_satisfied'])
        self.assertEqual([r['node_id'] for r in report['records']][0:5], ['root', 'ocr', 'rules', 'A', 'geometry'])
        self.assertIn('text.contrast', report['results'])

    def test_match_skips_later_computations(self):
        tree = demo_tree()
        tree['nodes'][3]['rule']['conditions'][0]['value'] = -1
        with patch('services.detection_compute.compute_gradient', side_effect=AssertionError('must not run')):
            report = execute_tree(self.content, tree, self.raw)
        self.assertEqual(report['matched_pattern'], 'A')
        self.assertEqual(report['stop_reason'], 'demo_pattern_matched')
        self.assertNotIn('image.gray', report['results'])
        self.assertTrue(all(r['status'] == 'skipped' for r in report['records'][4:]))

    def test_reorder_changes_execution_and_first_match(self):
        tree = demo_tree()
        tree['nodes'][1]['children'] = ['geometry', 'rules']
        tree['nodes'][5]['rule']['conditions'][0]['value'] = 0
        report = execute_tree(self.content, tree, self.raw)
        self.assertEqual(report['matched_pattern'], 'B')
        self.assertNotIn('text.rules', report['results'])

    def test_failed_dependency_and_independent_branch(self):
        with self.assertLogs('services.detection_tree', level='ERROR'):
            report = execute_tree(self.content, demo_tree())
        self.assertEqual(report['status'], 'incomplete')
        records = {r['node_id']: r for r in report['records']}
        self.assertEqual(records['A']['status'], 'blocked')
        self.assertEqual(records['gradient']['status'], 'completed')
        self.assertIsNone(report['matched_pattern'])

    def test_shared_computation_still_visits_children(self):
        tree = demo_tree()
        tree['nodes'][0]['children'].insert(1, 'ocr_again')
        tree['nodes'].append({'id': 'ocr_again', 'type': 'compute', 'operation': 'ocr.structured', 'children': []})
        calls = []
        def provider(image):
            calls.append(1)
            return self.raw
        report = execute_tree(self.content, tree, ocr_provider=provider)
        self.assertEqual(calls, [1])
        self.assertTrue(next(r for r in report['records'] if r['node_id'] == 'ocr_again')['reused'])

    def test_invalid_tree_before_execution(self):
        for mutate in (lambda t: t['nodes'][0]['children'].append('root'),
                       lambda t: t['nodes'][0].update(children=['gray', 'ocr']),
                       lambda t: t['nodes'].append({'id': 'orphan', 'type': 'group'}),
                       lambda t: t['nodes'][3].update(demo=False)):
            tree = demo_tree()
            mutate(tree)
            with self.assertRaises(ValueError):
                validate_tree(tree)

    def test_null_is_not_zero(self):
        rule = {'combine': 'all', 'conditions': [{'operation': 'text.geometry',
                'field': 'relative_box_height_mean', 'operator': 'lte', 'value': 0}]}
        self.assertIsNone(evaluate(rule, {'text.geometry': {'relative_box_height_mean': None}})[0])

    def test_api_persists_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Flask(__name__)
            app.config['DETECTION_RUN_STORE'] = directory
            app.register_blueprint(detection_bp)
            client = app.test_client()
            tree = client.get('/api/detection/trees/demo').json
            self.assertEqual(client.post('/api/detection/trees/validate', json=tree).status_code, 200)
            response = client.post('/api/detection/tree-runs', data={
                'file': (io.BytesIO(self.content), 'image.png'), 'tree': json.dumps(tree), 'ocr_input': self.raw})
            self.assertEqual(response.status_code, 201)
            stored = client.get('/api/detection/tree-runs/' + response.json['run_id'])
            self.assertEqual(stored.json, response.json)
            self.assertEqual(client.get('/api/detection/tree-runs/not-an-id').status_code, 404)
            self.assertEqual(client.post('/api/detection/trees/validate', json={}).status_code, 400)
