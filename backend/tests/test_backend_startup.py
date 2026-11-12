from pathlib import Path
import subprocess
import sys
import unittest


class BackendStartupTests(unittest.TestCase):
    def test_metadata_and_health_do_not_initialize_inference_runtime(self):
        backend = Path(__file__).resolve().parents[1]
        code = """
import sys
from app import create_app
client = create_app().test_client()
for path in ('/api/health', '/api/ocr/health', '/api/ocr/functions', '/api/whisper/health'):
    response = client.get(path)
    assert response.status_code == 200, (path, response.data)
assert client.post('/api/ocr/unload').status_code == 200
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert client.get('/api/health').json['device'] == 'not_initialized'
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=backend,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
