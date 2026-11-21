"""Lazy adapter from a normalized detection image to local DeepSeek grounding."""
from pathlib import Path
from tempfile import TemporaryDirectory


def recognize_grounding(image):
    # Reuse the existing model, inference lock and idle-unload lifecycle.
    from config import OCR_UPLOAD_FOLDER
    from .ocr_service import process_image

    with TemporaryDirectory(prefix='detection-', dir=OCR_UPLOAD_FOLDER) as directory:
        image_path = Path(directory) / 'image.png'
        image.save(image_path, format='PNG')
        result = process_image(
            image_path=str(image_path), function_type='markdown',
            resolution_mode='gundam', save_results=False,
            output_dir=str(Path(directory) / 'output'),
        )
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError('OCR returned no structured output')
        return result


# Automatic output is grounding even if the upload format selector says TSV.
# Custom providers without this attribute retain the existing format contract.
recognize_grounding.output_format = 'grounding'
