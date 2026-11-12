"""Small OpenAI-compatible server for the project-local Qwen model."""

import json
import threading
import time
import uuid
import urllib.request
from pathlib import Path
import config  # Initialize project-local model caches before inference imports.
from services.local_model_messages import normalize_messages, vision_messages

from flask import Flask, Response, jsonify, request
from flask_cors import CORS


MODEL_ID = "qwen2.5-3b-instruct"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_ID
VISION_MODEL_ID = "qwen2.5-vl-3b-instruct"
PRIVACY_MODEL_ID = "qwen3-4b-instruct-2507"
SUPPORTED_MODEL_IDS = (MODEL_ID, VISION_MODEL_ID, PRIVACY_MODEL_ID)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)
_tokenizer = None
_model = None
_processor = None
_loaded_id = None
torch = None  # Keep model listing/startup independent of heavy inference imports.
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def load_model(model_id=MODEL_ID):
    global _tokenizer, _model, _processor, _loaded_id, torch
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unknown local model: {model_id}")
    if torch is None:
        import torch as torch_runtime
        torch = torch_runtime
    if _model is not None and _loaded_id != model_id:
        _model = _tokenizer = _processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if _model is None:
        with _load_lock:
            if _model is None:
                try:
                    urllib.request.urlopen(
                        urllib.request.Request("http://127.0.0.1:5000/api/ocr/unload", method="POST"),
                        timeout=10,
                    ).read()
                except Exception:
                    pass
                if model_id == VISION_MODEL_ID:
                    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
                    model_dir = MODEL_DIR.parent / VISION_MODEL_ID
                    options = {'local_files_only': True, 'device_map': 'auto'}
                    if torch.cuda.is_available():
                        options['quantization_config'] = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                    else:
                        options['torch_dtype'] = torch.float32
                    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, **options)
                    _processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True,
                        min_pixels=256*28*28, max_pixels=768*28*28, use_fast=False)
                    _tokenizer = _processor.tokenizer
                elif model_id == PRIVACY_MODEL_ID:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                    model_dir = MODEL_DIR.parent / PRIVACY_MODEL_ID
                    options = {'local_files_only': True, 'device_map': 'auto', 'attn_implementation': 'sdpa'}
                    if torch.cuda.is_available():
                        options['quantization_config'] = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                    else:
                        options['torch_dtype'] = torch.float32
                    _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    _model = AutoModelForCausalLM.from_pretrained(model_dir, **options)
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
                    _model = AutoModelForCausalLM.from_pretrained(
                        MODEL_DIR, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        local_files_only=True, attn_implementation="eager",
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                _model.eval()
                _loaded_id = model_id
    return _tokenizer, _model


def unload_model():
    global _tokenizer, _model, _processor, _loaded_id
    with _generation_lock:
        _model = None
        _tokenizer = None
        _processor = None
        _loaded_id = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@app.post("/admin/unload")
def admin_unload():
    unload_model()
    return jsonify({"status": "ok", "loaded": False})


def generation_args(payload, streamer=None):
    model_id = payload.get('model', MODEL_ID)
    tokenizer, model = load_model(model_id)
    if model_id == VISION_MODEL_ID:
        from qwen_vl_utils import process_vision_info
        messages = vision_messages(payload.get('messages', []))
        prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = _processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors='pt').to(model.device)
    else:
        inputs = tokenizer.apply_chat_template(
            normalize_messages(payload.get("messages", [])), tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
    temperature = float(payload.get("temperature", 0.7) or 0)
    kwargs = {
        **inputs,
        "max_new_tokens": min(int(payload.get("max_tokens", 1024) or 1024), 8192),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update(temperature=max(temperature, 0.01), top_p=float(payload.get("top_p", 0.9)))
    if streamer is not None:
        kwargs["streamer"] = streamer
    return tokenizer, model, inputs, kwargs


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": _loaded_id, "loaded": _model is not None,
                    "device": ('not_initialized' if torch is None else 'cuda' if torch.cuda.is_available() else 'cpu')})


@app.get("/api/tags")
@app.get("/v1/models")
def models():
    if request.path.endswith("tags"):
        return jsonify({"models": [{"name": name, "model": name} for name in SUPPORTED_MODEL_IDS]})
    return jsonify({"object": "list", "data": [{"id": name, "object": "model"} for name in SUPPORTED_MODEL_IDS]})


@app.post("/v1/chat/completions")
def chat_completions():
    payload = request.get_json(force=True)
    model_id = payload.get('model', MODEL_ID)
    if model_id not in SUPPORTED_MODEL_IDS:
        return jsonify(error={'message': f'Unknown local model: {model_id}'}), 400
    try:
        if model_id == VISION_MODEL_ID:
            vision_messages(payload.get('messages', []))
        else:
            normalize_messages(payload.get('messages', []))
    except (ValueError, KeyError, OSError) as error:
        return jsonify(error={'message': str(error)}), 400
    request_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    if payload.get("stream", False):
        def generate_stream():
            with _generation_lock:
                tokenizer, model = load_model(model_id)
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                _, _, _, kwargs = generation_args(payload, streamer)
                errors = []
                def run_generation():
                    try:
                        with torch.inference_mode():
                            model.generate(**kwargs)
                    except Exception as error:
                        errors.append(str(error))
                        streamer.end()
                worker = threading.Thread(target=run_generation, daemon=True)
                worker.start()
                for text in streamer:
                    chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": model_id, "choices": [{"index": 0, "delta": {"content": text},
                             "finish_reason": None}]}
                    yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                worker.join()
                if errors:
                    yield "data: " + json.dumps({'error': {'message': errors[0]}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield "data: " + json.dumps({"id": request_id, "object": "chat.completion.chunk",
                      "created": created, "model": model_id, "choices": [{"index": 0, "delta": {},
                      "finish_reason": "stop"}]}, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
        return Response(generate_stream(), mimetype="text/event-stream")

    with _generation_lock:
        tokenizer, model, inputs, kwargs = generation_args(payload)
        with torch.inference_mode():
            output = model.generate(**kwargs)
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return jsonify({"id": request_id, "object": "chat.completion", "created": created,
                    "model": model_id, "choices": [{"index": 0, "message": {"role": "assistant",
                    "content": text}, "finish_reason": "stop"}]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=11434, threaded=True)
