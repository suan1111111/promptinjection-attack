"""Single-image Qwen2.5-VL-3B answer and RCS-style input representation.

Extraction follows sarendis56/Jailbreak_Detection_RCS/code/feature_extractor_qwen.py:
chat template -> multimodal processor -> forward(output_hidden_states=True)
-> last input token at the selected hidden-state index. No detector is trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse PrivaSee's offline cache setup, fixed model path, and result writer.
from test_qwen_vl import DEFAULT_MODEL_DIR, PROJECT_ROOT, save_results, status


def last_valid_token_index(attention_mask):
    """Find the last non-padding position, including with left padding."""
    positions = attention_mask[0].nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("输入没有有效 token")
    return int(positions[-1].item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="一张本地图片")
    parser.add_argument("question", help="文字问题")
    parser.add_argument("--layer", type=int, default=18,
                        help="hidden_states 索引：0=嵌入，1–36=解码层输出；默认18")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data/outputs/rcs_demo/result.json")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not args.question.strip():
        raise ValueError("问题不能为空")
    if not 0 <= args.layer <= 36:
        raise ValueError("Qwen2.5-VL-3B 的 --layer 必须在 0–36 之间")
    if args.max_new_tokens < 1 or args.max_visual_tokens < 256:
        raise ValueError("--max-new-tokens 必须为正数，--max-visual-tokens 至少为256")
    if not (DEFAULT_MODEL_DIR / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"找不到本地 Qwen2.5-VL-3B 权重：{DEFAULT_MODEL_DIR}")

    status("正在导入推理依赖……")
    import torch
    from PIL import Image, ImageOps
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("本演示复用现有 CUDA/NF4 路径，未检测到 CUDA")
    status(f"GPU：{torch.cuda.get_device_name(0)}；正在加载固定的 Qwen2.5-VL-3B……")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_DIR, local_files_only=True, device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ),
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        DEFAULT_MODEL_DIR, local_files_only=True, use_fast=False,
        min_pixels=256 * 28 * 28, max_pixels=args.max_visual_tokens * 28 * 28,
    )

    # Preserve PrivaSee's Windows path workaround: detached PIL image, no file URI.
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    token_index = last_valid_token_index(inputs.attention_mask)
    token_id = int(inputs.input_ids[0, token_index].item())

    status(f"提取输入阶段 hidden_states[{args.layer}]，token 位置 {token_index}……")
    with torch.inference_mode():
        # Same extraction method as upstream. Disable KV cache for this separate
        # input-only forward; generation below starts fresh with the same inputs.
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        vector = outputs.hidden_states[args.layer][0, token_index, :].detach().float().cpu()
        del outputs
    if not torch.isfinite(vector).all().item():
        raise RuntimeError("隐藏向量出现 NaN/Inf")
    hidden_summary = {
        "layer_index": args.layer,
        "layer_convention": "0=embedding; 1..36=decoder outputs; 36 includes final RMSNorm",
        "stage": "input_forward_before_generation",
        "input_token_count": int(inputs.attention_mask[0].sum().item()),
        "token_index": token_index,
        "token_id": token_id,
        "token_text": processor.tokenizer.decode([token_id], skip_special_tokens=False),
        "dimension": vector.numel(),
        "first_five": vector[:5].tolist(),
        "l2_norm": float(torch.linalg.vector_norm(vector).item()),
    }
    status("输入向量已提取，正在生成回答……")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, output_hidden_states=False,
                                   return_dict_in_generate=False)
    answer = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    output_path = args.output.expanduser().resolve()
    save_results(output_path, {
        "image": str(image_path), "question": args.question, "answer": answer,
        "model": str(DEFAULT_MODEL_DIR),
        "quantization": "bitsandbytes NF4, double quantization, BF16 compute",
        "hidden_state": hidden_summary,
    })
    print(f"\n模型回答：\n{answer}")
    print(f"\n输入隐藏向量：hidden_states[{args.layer}]")
    print(f"最后有效 token：位置={token_index}，ID={token_id}，文本={hidden_summary['token_text']!r}")
    print(f"维度：{hidden_summary['dimension']}")
    print(f"前五个数值：{hidden_summary['first_five']}")
    print(f"L2 范数：{hidden_summary['l2_norm']:.6f}")
    print(f"结果已保存：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("操作已取消。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
