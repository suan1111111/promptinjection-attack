import unittest
from flask import Flask
from services.detection_api import detection_bp
from services.detection_catalog import get_catalog, validate_catalog


class DetectionCatalogTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(detection_bp)
        self.client = app.test_client()

    def test_discovery_and_details_agree(self):
        response = self.client.get('/api/detection/capabilities')
        self.assertEqual(response.status_code, 200)
        catalog = response.get_json()
        validate_catalog(catalog['operations'])
        self.assertTrue(catalog['execution_available'])
        for operation in catalog['operations']:
            detail = self.client.get('/api/detection/capabilities/' + operation['id'])
            self.assertEqual(detail.get_json()['operation'], operation)

    def test_unknown_operation(self):
        response = self.client.get('/api/detection/capabilities/missing')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()['error']['code'], 'operation_not_found')

    def test_dependencies_do_not_force_unrelated_calculations(self):
        operations = {op['id']: op for op in get_catalog()['operations']}
        self.assertEqual(operations['text.geometry']['dependencies'], ['ocr.structured'])
        self.assertEqual(operations['text.gradient_overlap']['dependencies'],
                         ['ocr.structured', 'image.gradient'])
        self.assertTrue(all(op['executable'] for op in operations.values()))

    def test_invalid_graphs_rejected(self):
        for graph in ([{'id': 'a', 'dependencies': ['missing']}],
                      [{'id': 'a', 'dependencies': ['b']}, {'id': 'b', 'dependencies': ['a']}],
                      [{'id': 'a', 'dependencies': []}, {'id': 'a', 'dependencies': []}]):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_catalog(graph)

    def test_callers_cannot_mutate_catalog(self):
        catalog = get_catalog()
        catalog['operations'][0]['dependencies'].append('bad')
        validate_catalog(get_catalog()['operations'])


if __name__ == '__main__':
    unittest.main()
