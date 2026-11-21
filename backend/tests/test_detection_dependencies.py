import io
import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from flask import Flask
from services.detection_api import detection_bp
from services.detection_tree import TreeDependencyError, validate_tree, demo_tree


def flat_tree(operations):
    nodes = [{'id': str(i), 'type': 'compute', 'operation': op, 'children': []}
             for i, op in enumerate(operations)]
    return {'schema_version': 1, 'root_id': 'root', 'nodes': [
        {'id': 'root', 'type': 'group', 'children': [n['id'] for n in nodes]}, *nodes]}


class DependencyTests(unittest.TestCase):
    def test_independent_reordering_is_allowed_without_mutation(self):
        for ops in (['ocr.structured', 'image.gray', 'image.gradient', 'text.gradient_overlap'],
                    ['image.gray', 'image.gradient', 'ocr.structured', 'text.gradient_overlap']):
            tree = flat_tree(ops)
            before = deepcopy(tree)
            self.assertEqual(validate_tree(tree), ['root', '0', '1', '2', '3'])
            self.assertEqual(tree, before)

    def test_collect_all_missing_and_late_dependencies(self):
        tree = flat_tree(['text.gradient_overlap', 'ocr.structured'])
        with self.assertRaises(TreeDependencyError) as caught:
            validate_tree(tree)
        issues = {i['dependency_operation_id']: i for i in caught.exception.issues}
        self.assertEqual(issues['ocr.structured']['reason'], 'not_computed_before')
        self.assertEqual(issues['ocr.structured']['producer_node_ids'], ['1'])
        self.assertEqual(issues['image.gradient']['reason'], 'missing_computation')
        self.assertEqual(issues['image.gradient']['producer_node_ids'], [])
        self.assertTrue(all(i['node_id'] == '0' for i in issues.values()))

    def test_pattern_and_completion_dependencies(self):
        tree = flat_tree(['text.rules', 'ocr.structured'])
        pattern = deepcopy(demo_tree()['nodes'][3])
        tree['nodes'].append(pattern)
        tree['nodes'][0]['children'].insert(0, 'A')
        tree['completion_operations'] = ['text.contrast', 'image.gray']
        with self.assertRaises(TreeDependencyError) as caught:
            validate_tree(tree)
        issues = {i['node_id']: i for i in caught.exception.issues}
        self.assertEqual(issues['A']['dependency_operation_id'], 'text.rules')
        self.assertEqual(issues['completion:0']['producer_node_ids'], ['completion:1'])

    def test_repeated_producer_can_satisfy_dependency(self):
        tree = flat_tree(['ocr.structured', 'text.rules', 'ocr.structured'])
        self.assertEqual(len(validate_tree(tree)), 4)

    def test_both_endpoints_reject_before_computation(self):
        app = Flask(__name__)
        app.register_blueprint(detection_bp)
        tree = flat_tree(['text.rules', 'ocr.structured'])
        client = app.test_client()
        with patch('services.detection_tree.ComputationSession', side_effect=AssertionError('must not execute')):
            validation = client.post('/api/detection/trees/validate', json=tree)
            execution = client.post('/api/detection/tree-runs', data={
                'tree': json.dumps(tree), 'file': (io.BytesIO(b'invalid image'), 'sample.png')})
        self.assertEqual(validation.status_code, 400)
        self.assertEqual(execution.status_code, 400)
        self.assertEqual(validation.json['error'], execution.json['error'])
        self.assertEqual(validation.json['error']['code'], 'invalid_dependency_order')
        self.assertEqual(validation.json['error']['issues'][0]['node_id'], '0')
