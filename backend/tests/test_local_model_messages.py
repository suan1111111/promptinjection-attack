import base64
import io
from pathlib import Path
import sys
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.local_model_messages import normalize_messages, vision_messages


class MessageTests(unittest.TestCase):
    def test_picture_preserved_and_detached(self):
        stream = io.BytesIO()
        Image.new('RGB', (7, 9), (12, 34, 56)).save(stream, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'describe'}, {'type': 'image_url', 'image_url': {'url': url}}]}]
        output = vision_messages(messages)
        image = output[0]['content'][1]['image']
        self.assertEqual(image.size, (7, 9))
        self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(output[0]['content'][0]['text'], 'describe')
        with self.assertRaises(ValueError):
            normalize_messages(messages)

    def test_remote_and_file_urls_rejected(self):
        for url in ('file:///D:/example.png', 'https://example.com/image.png'):
            with self.assertRaises(ValueError):
                vision_messages([{'content': [{'type': 'image_url', 'image_url': {'url': url}}]}])

    def test_text_history_keeps_roles(self):
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'assistant', 'content': 'history'}]
        self.assertEqual(normalize_messages(messages), messages)
        self.assertEqual(vision_messages(messages), messages)


if __name__ == '__main__':
    unittest.main()
