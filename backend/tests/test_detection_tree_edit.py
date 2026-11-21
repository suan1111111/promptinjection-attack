import io
import unittest
from copy import deepcopy
from flask import Flask
from PIL import Image

from services.detection_api import detection_bp
from services.detection_tree import demo_tree, execute_tree, validate_tree, TreeDependencyError
from services.detection_tree_edit import add_node


class TreeEditTests(unittest.TestCase):
    def test_insert_preserves_existing_tree_and_reuses_calculation(self):
        tree = demo_tree()
        before = deepcopy(tree)
        node = {'id': 'ocr-again', 'type': 'compute', 'operation': 'ocr.structured'}
        result = add_node(tree, 'ocr', node, 0)
        self.assertEqual(tree, before)
        self.assertNotIn('children', node)
        self.assertEqual(result['tree']['nodes'][1]['children'], ['ocr-again', 'rules', 'geometry'])
        image = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(image, 'PNG')
        calls = []
        def provider(image):
            calls.append(1)
            return '<|ref|>text<|/ref|><|det|>[[100,100,500,500]]<|/det|>hello'
        report = execute_tree(image.getvalue(), result['tree'], ocr_provider=provider)
        self.assertEqual(len(calls), 1)
        self.assertTrue(next(r for r in report['records'] if r['node_id'] == 'ocr-again')['reused'])

    def test_add_demo_judgment_and_validate(self):
        node = deepcopy(demo_tree()['nodes'][3])
        node.pop('id')
        result = add_node(demo_tree(), 'rules', node)
        self.assertEqual(result['index'], 1)
        self.assertIn(result['node_id'], validate_tree(result['tree']))
        self.assertTrue(result['requires_validation'])

    def test_allow_unresolved_draft_dependencies_but_not_final_validation(self):
        result = add_node(demo_tree(), 'root', {
            'type': 'compute', 'operation': 'text.contrast'}, 0)
        with self.assertRaises(TreeDependencyError):
            validate_tree(result['tree'])
        # The next edit must also accept an unfinished draft.
        result = add_node(result['tree'], 'root', {'type': 'compute', 'operation': 'image.gray'}, 0)
        self.assertEqual(result['status'], 'draft')

    def test_invalid_edits_leave_input_untouched(self):
        tree = demo_tree()
        before = deepcopy(tree)
        valid = {'type': 'compute', 'operation': 'image.gray'}
        for parent, node, index in [
            ('missing', valid, 0), ('A', valid, 0), ('root', valid, -1),
            ('root', valid, True), ('root', valid, 9),
            ('root', {**valid, 'id': 'root'}, 0),
            ('root', {**valid, 'operation': 'missing'}, 0),
            ('root', {**valid, 'children': ['ocr']}, 0),
            ('root', {'type': 'pattern', 'demo': True}, 0),
        ]:
            with self.subTest(parent=parent, node=node, index=index):
                with self.assertRaises(ValueError):
                    add_node(tree, parent, node, index)
                self.assertEqual(tree, before)

    def test_endpoint_returns_draft_and_validation_stays_strict(self):
        app = Flask(__name__)
        app.register_blueprint(detection_bp)
        client = app.test_client()
        response = client.post('/api/detection/trees/nodes', json={
            'tree': demo_tree(), 'parent_id': 'root', 'index': 0,
            'node': {'type': 'compute', 'operation': 'text.contrast'}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'draft')
        checked = client.post('/api/detection/trees/validate', json=response.json['tree'])
        self.assertEqual(checked.status_code, 400)
        self.assertEqual(client.post('/api/detection/trees/nodes', json=[]).status_code, 400)
