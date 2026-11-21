from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import tempfile
import unittest

from flask import Flask
from services.detection_api import detection_bp
from services.detection_tree import demo_tree, validate_tree, TreeDependencyError
from services.detection_tree_edit import delete_node
from services.detection_tree_store import TreeStore, VersionConflict


class TreeStorageTests(unittest.TestCase):
    def test_delete_subtree_and_promote_children(self):
        tree = demo_tree()
        original = deepcopy(tree)
        deleted = delete_node(tree, 'ocr', 'subtree')
        self.assertEqual(set(deleted['removed_node_ids']), {'ocr', 'rules', 'A', 'geometry', 'B'})
        self.assertEqual(deleted['tree']['nodes'][0]['children'], ['gray'])
        with self.assertRaises(TreeDependencyError):
            validate_tree(deleted['tree'])
        promoted = delete_node(tree, 'ocr', 'promote_children')
        self.assertEqual(promoted['tree']['nodes'][0]['children'], ['rules', 'geometry', 'gray'])
        self.assertEqual(promoted['removed_node_ids'], ['ocr'])
        self.assertEqual(tree, original)
        self.assertEqual(delete_node(tree, 'A', 'promote_children')['tree']['nodes'][2]['children'], [])
        for node, mode in [('root', 'subtree'), ('missing', 'subtree'), ('ocr', 'unknown')]:
            with self.assertRaises(ValueError):
                delete_node(tree, node, mode)

    def test_persistence_publication_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TreeStore(directory)
            payload = {'name': 'My tree', 'status': 'published', 'tree': demo_tree()}
            first = store.save(payload)
            self.assertEqual(TreeStore(directory).read(first['id']), first)
            draft = delete_node(first['tree'], 'ocr', 'subtree')['tree']
            second = store.save({**payload, 'tree': draft, 'status': 'draft', 'expected_version': 1}, first['id'])
            self.assertEqual(second['published_tree'], first['tree'])
            self.assertEqual(second['published_version'], 1)
            self.assertEqual(second['version'], 2)
            with self.assertRaises(TreeDependencyError):
                store.save({**payload, 'tree': draft, 'expected_version': 2}, first['id'])
            with self.assertRaises(VersionConflict):
                store.save({**payload, 'expected_version': 1}, first['id'])
            self.assertEqual(store.read(first['id']), second)
            self.assertEqual(store.list()[0]['id'], first['id'])
            self.assertNotIn('tree', store.list()[0])
            with self.assertRaises(ValueError):
                store.read('../outside')

    def test_concurrent_update_has_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {'name': 'Tree', 'status': 'draft', 'tree': demo_tree()}
            first = TreeStore(directory).save(payload)
            def update(_):
                try:
                    TreeStore(directory).save({**payload, 'expected_version': 1}, first['id'])
                    return 'saved'
                except VersionConflict:
                    return 'conflict'
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertCountEqual(list(pool.map(update, range(2))), ['saved', 'conflict'])

    def test_api_save_read_update_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Flask(__name__)
            app.config['DETECTION_TREE_STORE'] = directory
            app.register_blueprint(detection_bp)
            client = app.test_client()
            payload = {'name': 'Saved tree', 'status': 'published', 'tree': demo_tree()}
            created = client.post('/api/detection/tree-configs', json=payload)
            self.assertEqual(created.status_code, 201)
            url = '/api/detection/tree-configs/' + created.json['id']
            self.assertEqual(client.get(url).json, created.json)
            self.assertEqual(len(client.get('/api/detection/tree-configs').json['items']), 1)
            deleted = client.post('/api/detection/trees/nodes/delete', json={
                'tree': payload['tree'], 'node_id': 'ocr', 'mode': 'subtree'})
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.get(url).json, created.json)
            updated = client.put(url, json={**payload, 'status': 'draft',
                'tree': deleted.json['tree'], 'expected_version': 1})
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(client.put(url, json={**payload, 'expected_version': 1}).status_code, 409)
            self.assertEqual(client.put(url, json={**payload, 'tree': deleted.json['tree'],
                'expected_version': 2}).status_code, 400)
            self.assertEqual(client.get('/api/detection/tree-configs/' + '0' * 32).status_code, 404)
            self.assertEqual(client.post('/api/detection/tree-configs', json=[]).status_code, 400)
