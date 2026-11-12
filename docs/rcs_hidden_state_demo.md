# RCS 输入隐藏状态最小演示

本演示固定使用项目本地 `models/qwen2.5-vl-3b-instruct`，接受一张图片和文字问题，输出模型回答，以及指定层最后一个有效输入 token 的向量维度、前五个数值和 L2 范数。无需数据集或检测器训练。

在 PrivaSee 根目录的 PowerShell 中运行：

```powershell
$env:PYTHONUTF8='1'
& '.\.venv\Scripts\python.exe' backend/demo_rcs_hidden_state.py picture/medical-page-09.png '请简短描述图片中的内容。' --layer 18
```

`--layer` 默认18，范围0–36，与 Transformers 的 `outputs.hidden_states` 索引一致：0为嵌入输出，1–36对应解码层输出，36含最终 RMSNorm。选层只是演示参数，未经训练或评估，不代表论文的最佳安全层。

可选参数：`--max-new-tokens`（默认128）、`--max-visual-tokens`（默认768，至少256）、`--output`（默认 `data/outputs/rcs_demo/result.json`，再次运行会覆盖）。

## 方法与来源

参考 RCS 官方 [feature_extractor_qwen.py](https://github.com/sarendis56/Jailbreak_Detection_RCS/blob/main/code/feature_extractor_qwen.py) 的 `HiddenStateExtractor.extract_hidden_states` 多模态流程：

1. 图片和问题构造用户消息，`apply_chat_template(..., add_generation_prompt=True)`。
2. `process_vision_info` 和模型 processor 构造多模态输入。
3. `model(..., output_hidden_states=True)` 执行输入前向计算。
4. 从 `outputs.hidden_states[layer]` 提取最后输入位置的向量。

针对 PrivaSee 的调整：复用 `test_qwen_vl.py` 的离线缓存配置、固定模型目录与 JSON 保存函数；沿用其 NF4 四位量化、BF16计算及视觉 token 预算；使用 PIL 图片避免 Windows 的 file URI 问题；从 attention mask 的最后一个非零位置取 token，兼容左/右 padding；转 FP32 后计算数值和范数。输入前向关闭 KV cache，释放全层输出后再用同一份输入调用 `generate`，因此会有两次输入前向计算，但不会保存全部生成步骤的隐藏状态。

“最后一个有效输入 token”指完整聊天模板的末尾，可能是换行或回答起始提示，并非问题最后一个字，也不是生成回答的末尾。结果记录实际 token 索引、ID、文本与输入有效 token 数。

本演示保留当前项目环境，不安装官方实验的整套依赖。需要可用 CUDA 和现有本地权重。NF4量化、图像预算以及所选层都会影响向量数值，因此这些结果用于展示提取流程，不用于声称复现论文检测指标。
