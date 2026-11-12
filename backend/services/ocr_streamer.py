"""Loaded only when OCR generation needs a Transformers streamer."""
from transformers import TextStreamer

class CallbackTextStreamer(TextStreamer):
    """自定义 TextStreamer，通过回调函数实时发送生成的文本"""
    def __init__(self, tokenizer, callback=None, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.callback = callback
        self.text_queue = []
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """当文本块生成完成时调用"""
        # 过滤掉 EOS token
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        if eos_text in text:
            text = text.replace(eos_text, '')
        if text and self.callback:
            self.callback(text)

