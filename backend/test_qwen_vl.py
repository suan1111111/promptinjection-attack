"""Test local images or image-based PDFs with Qwen2.5-VL-3B-Instruct.

The official BF16 checkpoint is quantized to NF4 by bitsandbytes while loading.
The model is loaded once from disk; selected PDF pages are analyzed together.
Interactive follow-up questions reuse the model and retain conversation history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-vl-3b-instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "qwen_vl"
POPPLER_BIN = PROJECT_ROOT / "tools" / "poppler" / "Library" / "bin"
TEMP_DIR = PROJECT_ROOT / ".tmp"

# Set these before importing Hugging Face libraries. Loading stays offline and
# all configurable caches remain on the project drive.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".hf-cache" / "modules"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PROJECT_ROOT / ".hf-cache" / "hub"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


def status(message: str) -> None:
    """Print progress immediately, even when VS Code buffers terminal output."""
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask local Qwen2.5-VL-3B questions about an image or PDF."
    )
    parser.add_argument("input_file", nargs="?", help="Path to an image or PDF")
    parser.add_argument(
        "question",
        nargs="?",
        default="请客观描述当前页面的内容，并识别其中的文字。",
        help="Question asked for every image or PDF page",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, help="Result JSON file")
    parser.add_argument(
        "--pages", help='PDF pages, for example "1-3,5,8" (default: all pages)'
    )
    parser.add_argument("--dpi", type=int, default=144, help="PDF render DPI")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--load-dtype", choices=("float16", "bfloat16"), default="float16",
        help="Non-quantized model dtype; use bfloat16 to reproduce the pixel trainer (default: float16)",
    )
    parser.add_argument(
        "--min-visual-tokens",
        type=int,
        default=256,
        help="Minimum visual-token budget per image (default: 256)",
    )
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--once", action="store_true", help="Answer once and exit")
    return parser.parse_args()


def get_input_path(value: str | None) -> Path:
    if not value:
        value = input("请输入图片或 PDF 的完整路径: ").strip().strip('"')
    input_path = Path(value).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    return input_path


def parse_page_selection(value: str | None, page_count: int) -> list[int]:
    if not value:
        return list(range(1, page_count + 1))
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"页码范围无效: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise ValueError(f"页码超出 1-{page_count} 范围: {invalid}")
    if not pages:
        raise ValueError("--pages 没有包含有效页码")
    return sorted(pages)


def save_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)


def ask_about_images(
    image_paths: list[Path],
    question: str,
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    max_new_tokens: int,
    history: list[dict[str, Any]] | None = None,
) -> str:
    from PIL import Image, ImageOps

    # qwen-vl-utils 0.0.14 handles file:// URIs incorrectly on Windows by
    # stripping "file://" and leaving a Unix-style "/D:/..." path. Passing a
    # detached PIL image avoids that path conversion for images and PDF pages.
    images = []
    for image_path in ([] if history else image_paths):
        with Image.open(image_path) as source_image:
            images.append(ImageOps.exif_transpose(source_image).convert("RGB"))

    content: list[dict[str, Any]] = []
    if len(images) > 1:
        content.append({
            "type": "text",
            "text": "以下图片按给出的顺序排列，请结合全部图片回答最后的问题。",
        })
    for position, image in enumerate(images, start=1):
        if len(images) > 1:
            content.append({"type": "text", "text": f"第 {position} 张图片："})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    messages = list(history or []) + [{
        "role": "user",
        "content": content,
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generated_ids, trimmed_ids
    if history is not None:
        history[:] = messages + [{
            "role": "assistant", "content": [{"type": "text", "text": answer}],
        }]
    return answer


def chat_about_images(
    image_paths: list[Path], args: argparse.Namespace, model: Any,
    processor: Any, process_vision_info: Any, torch: Any,
    payload: dict[str, Any], output_path: Path, result_metadata: dict[str, Any],
) -> None:
    history: list[dict[str, Any]] = []
    question = args.question
    session = 1
    if not args.once:
        print("连续对话已开启：输入问题继续追问；/clear 清空上下文；/exit 退出。")
    while True:
        if question:
            answer = ask_about_images(
                image_paths, question, model, processor, process_vision_info,
                torch, args.max_new_tokens, history=history,
            )
            payload["results"].append({
                **result_metadata, "session": session,
                "question": question, "answer": answer,
            })
            save_results(output_path, payload)
            print(f"\n模型回答：\n{answer}", flush=True)
        if args.once:
            break
        try:
            question = input("\n你：").strip()
        except EOFError:
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if question.lower() == "/clear":
            history.clear()
            session += 1
            question = ""
            print("上下文已清空，当前图片或 PDF 保持不变。")


def main() -> int:
    status("程序已启动，正在检查输入文件和模型目录……")
    args = parse_args()
    input_path = get_input_path(args.input_file)
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"模型目录不完整: {model_dir}")
    if (
        args.max_new_tokens < 1
        or args.min_visual_tokens < 4
        or args.max_visual_tokens < 4
    ):
        raise ValueError("token 数量参数必须大于 0")
    if args.min_visual_tokens > args.max_visual_tokens:
        raise ValueError("--min-visual-tokens 不能大于 --max-visual-tokens")
    if args.dpi < 72 or args.dpi > 300:
        raise ValueError("--dpi 应在 72 到 300 之间")

    status("正在导入 PyTorch（首次导入可能需要 1–2 分钟）……")
    import torch
    status("PyTorch 导入完成，正在导入 Qwen/Transformers 组件……")
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor, BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    status("运行依赖导入完成。")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 未检测到 CUDA；请检查 NVIDIA 驱动和 PyTorch。")

    status(f"GPU: {torch.cuda.get_device_name(0)}")
    status(f"模型: {model_dir}")
    status("正在以 bitsandbytes NF4 加载模型，通常需要约两分钟……")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, quantization_config=quantization_config,
        device_map="auto", low_cpu_mem_usage=True, local_files_only=True,
        torch_dtype=getattr(torch, args.load_dtype),
    )
    model.eval()
    status("模型权重加载完成，正在加载图像处理器……")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
        use_fast=False,
        local_files_only=True,
    )
    status("模型和图像处理器均已就绪。")

    output_path = (
        args.output or DEFAULT_OUTPUT_DIR / f"{input_path.stem}_results.json"
    ).resolve()
    payload: dict[str, Any] = {
        "input_file": str(input_path),
        "model": str(model_dir),
        "quantization": "bitsandbytes NF4, double quantization, BF16 compute",
        "load_dtype": args.load_dtype,
        "question": args.question,
        "results": [],
    }

    if input_path.suffix.lower() != ".pdf":
        print("模型已加载，正在分析图片……")
        chat_about_images(
            [input_path], args, model, processor, process_vision_info,
            torch, payload, output_path, {"image": str(input_path)},
        )
    else:
        from pdf2image import convert_from_path, pdfinfo_from_path
        if not (POPPLER_BIN / "pdftoppm.exe").is_file():
            raise FileNotFoundError(f"找不到 Poppler: {POPPLER_BIN}")
        page_count = int(pdfinfo_from_path(
            input_path, poppler_path=str(POPPLER_BIN)
        )["Pages"])
        pages = parse_page_selection(args.pages, page_count)
        print(f"PDF 共 {page_count} 页，本次按顺序共同分析 {len(pages)} 页: {pages}")
        if len(pages) > 8:
            raise ValueError(
                "一次共同输入超过 8 页，容易耗尽显存；请用 --pages 选择不超过 8 页。"
            )
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="qwen_pdf_", dir=TEMP_DIR) as temp_dir:
            page_images: list[Path] = []
            for position, page_number in enumerate(pages, start=1):
                print(f"[{position}/{len(pages)}] 正在渲染第 {page_number} 页……", flush=True)
                rendered = convert_from_path(
                    input_path, dpi=args.dpi,
                    first_page=page_number, last_page=page_number,
                    fmt="png", output_folder=temp_dir, paths_only=True,
                    poppler_path=str(POPPLER_BIN), thread_count=1,
                )
                page_images.append(Path(rendered[0]))

            status(
                f"页面已按 {pages} 的顺序准备完成，正在作为同一次请求进行分析……"
            )
            chat_about_images(
                page_images, args, model, processor, process_vision_info,
                torch, payload, output_path, {"pages": pages},
            )
            torch.cuda.empty_cache()

    save_results(output_path, payload)
    print(f"\n结果已保存: {output_path}")
    print(f"当前模型显存占用约: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n操作已取消。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n运行失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
