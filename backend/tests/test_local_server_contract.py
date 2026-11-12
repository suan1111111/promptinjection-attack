from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_llm_server as server


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_listing_does_not_load_models(self):
        self.assertEqual({item['id'] for item in self.client.get('/v1/models').json['data']},
                         set(server.SUPPORTED_MODEL_IDS))
        self.assertIsNone(server.torch)
        self.assertFalse(self.client.get('/health').json['loaded'])

    def test_unknown_model_returns_error(self):
        result = self.client.post('/v1/chat/completions', json={'model': 'not-installed'})
        self.assertEqual(result.status_code, 400)
        self.assertIn('Unknown', result.json['error']['message'])

    def test_text_model_does_not_silently_drop_image(self):
        result = self.client.post('/v1/chat/completions', json={
            'model': server.MODEL_ID,
            'messages': [{'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]})
        self.assertEqual(result.status_code, 400)
        self.assertIn('vision model', result.json['error']['message'])


if __name__ == '__main__':
    unittest.main()
