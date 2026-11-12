import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.ocr_inference import infer_with_streamer


class StockModel:
    def generate(self, **kwargs):
        stream = kwargs.get('streamer')
        if stream is not None:
            stream.append('first chunk')
        if kwargs.get('fail'):
            raise RuntimeError('generation failed')
        return 'finished'

    def infer(self, tokenizer, **kwargs):
        return self.generate(**kwargs)


class StreamCompatibilityTests(unittest.TestCase):
    def test_stock_streams_during_generation_and_restores_method(self):
        model = StockModel()
        chunks = []
        result = infer_with_streamer(model, None, streamer=chunks)
        self.assertEqual(chunks, ['first chunk'])
        self.assertEqual(result, 'finished')
        self.assertNotIn('generate', model.__dict__)

    def test_exception_restores_generate(self):
        model = StockModel()
        with self.assertRaises(RuntimeError):
            infer_with_streamer(model, None, streamer=[], fail=True)
        self.assertNotIn('generate', model.__dict__)

    def test_upstream_custom_streamer_contract(self):
        class CustomModel:
            def infer(self, tokenizer, custom_streamer=None, **kwargs):
                custom_streamer.append('upstream')
                return 'done'
        chunks = []
        self.assertEqual(infer_with_streamer(CustomModel(), None, streamer=chunks), 'done')
        self.assertEqual(chunks, ['upstream'])


if __name__ == '__main__':
    unittest.main()
