# PrivaSee Backend Services
# 模块化服务包

__all__ = ['ocr_bp', 'whisper_bp']


def __getattr__(name):
    # Lightweight feature extraction must not load Torch/OCR/Whisper.
    if name == 'ocr_bp':
        from .ocr_service import ocr_bp
        return ocr_bp
    if name == 'whisper_bp':
        from .whisper_service import whisper_bp
        return whisper_bp
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
