# RCS 完整特征提取与投影训练

代码与默认输出均位于 D 盘 PrivaSee。模型固定为本地 Qwen2.5-VL-3B-Instruct，沿用 NF4 双重量化、BF16计算，不安装新环境、不下载模型和数据集。

## 来源与复用范围

官方仓库：https://github.com/sarendis56/Jailbreak_Detection_RCS

固定提交：`3ea8e29f3542e7962a7eaed7996dc5654b4035fd`。

- `backend/third_party/rcs/kcd.py.txt`、`feature_extractor_qwen.py.txt` 是该提交的原始源码快照，仅作来源留档，不直接导入整个实验脚本。
- `backend/rcs_projection_core.py` 中的 `ProjectionConfig`、`LearnedProjection`、`compute_contrastive_loss` 从官方 kcd.py 原样摘取。独立模块仅增加必要 torch 导入和 Qwen 配置，避免官方脚本顶层 LLaVA 依赖、GPU初始化和数据集加载。测试比较 AST，检查三个定义没有改动。
- `rcs_features.py` 沿用官方的消息模板 → process_vision_info → processor → 输入前向 `output_hidden_states=True` → 最后输入token向量。适配现有离线 NF4 加载、Windows PIL路径、attention mask最后非零位置、FP32保存。一次前向提取所选所有层，关闭KV cache，不生成回答，不训练模型。
- 完整清单、逐样本原子缓存、内容校验、失败重试及标签检查是本项目工程补充，不是声称作者原样提供这些接口。
- `rcs_projection.py` 训练循环保留作者 Adam（weight_decay=1e-4）、余弦调度（最小学习率0.05倍）、梯度裁剪1、shuffle、drop_last=True、15轮训练loss不改善早停。与作者相同返回最后一轮模型，不自动恢复最小loss轮。工程补充为配置/标签校验和checkpoint保存加载。

## 输入清单

UTF-8 JSON数组，图片路径相对于清单文件所在目录。每条记录：

| 字段 | 规则 |
|---|---|
| id | 唯一非空字符串；不要用问题本身作为ID |
| image | 一张本地图片路径 |
| question | 非空文字问题 |
| label | 0良性、1恶意；未标注为null，不自动补为0 |
| dataset | 数据来源／攻击来源名称；不知道为null |
| split | train、val、test或unassigned；默认unassigned |

可选额外字段会随记录保存。提取不要求安全标签；投影训练只读取split=train，并要求其安全、来源标签完整。train标签不能从“图片经过扰动”或训练loss下降自动推断。

现有 `examples/rcs_inputs.json` 只提供一条未标注样本用于提取检查，不是训练集。准备真实训练清单时复制后补充真实图片、问题和经核验标签。验证/测试与训练的图片问题对必须分离；同原图的不同扰动版本也应按原图分组划分，程序不能自动判定两个文件是否来自同一原图。

## 提取：从项目根目录运行

```powershell
$env:PYTHONUTF8='1'
# 仅检查清单，不加载模型、不写结果
& '.\.venv\Scripts\python.exe' -B backend/rcs_features.py --manifest examples/rcs_inputs.json --output data/outputs/rcs_features --layers 12,18,24 --validate-only

# 多层完整向量；每条输入一次前向，模型在整个清单中只加载一次
& '.\.venv\Scripts\python.exe' -B backend/rcs_features.py --manifest examples/rcs_inputs.json --output data/outputs/rcs_features --layers 12,18,24
```

`--layers all` 保存0..36共37个隐藏状态索引；0为嵌入，1..36为解码层表示，36包含最终RMSNorm。候选层不是自动选层结果。

最后输入token含聊天模板，可能是换行，既非回答末尾也不一定是问题最后一个字。所有层使用同一个token位置。`--min-visual-tokens`默认256，`--max-visual-tokens`默认768，改变预处理必须换输出目录。

## 输出、恢复与失败

输出目录包括：

- `config.json`：模型与处理器身份、层号、运行版本、视觉预算及配置指纹。
- `features/<id哈希>.npz`：`vectors`为float32的`[层数,2048]`完整矩阵；`metadata`为JSON字符串，含层号、图片哈希、原记录、token位置/ID/文本、完整input_ids、实际视觉网格和prompt哈希。读取使用`allow_pickle=False`。
- `index.json`：按本次清单顺序逐条记录completed/failed及错误信息；每完成一条原子更新。

再次运行相同命令：核对输入内容与配置，跳过成功记录，重试失败或损坏的记录；即使中断在index更新前，也可识别已写好的单条NPZ。修改同ID的问题/标签/图片后重新提取该条；层号或运行配置改变则拒绝混写。缺少图片等单条错误不终止后续记录，最后以非零退出码报告有失败。模型加载/CUDA问题也会记录为失败。

同一个输出目录只运行一个提取进程。缓存恢复发生在“样本之间”，不恢复一次前向内部进度。重新运行时index对应当前清单，目录中历史但不在当前清单内的NPZ不会参与训练。

读取完整特征的Python示例（从backend可导入）：

```python
from rcs_features import load_feature
vectors, metadata = load_feature('D:/CodexProjects/PrivaSee/data/outputs/rcs_features/features/实际文件名.npz')
layer18 = vectors[metadata['layers'].index(18)]  # shape (2048,)
```

## 投影网络与训练接口

Qwen结构：2048 → 512 → 256 → 256；保留作者BatchNorm、ReLU、Dropout=0.3及Xavier初始化。损失内部先L2归一化；来源结构项权重1，安全分离项权重5，包含作者的类内紧凑项。

```powershell
# 收集、标注并提取真实训练集以后才运行；每层单独训练，串行执行
& '.\.venv\Scripts\python.exe' -B backend/rcs_projection.py --features data/outputs/rcs_features --output data/outputs/rcs_projection/run01 --layers 12,18,24 --epochs 100 --batch-size 64 --device cpu
```

默认100轮、batch_size=64、学习率1e-3、seed=42；可指定`--device cuda`。这里只加载缓存向量和小网络，不加载Qwen。训练样本不足一个batch会明确拒绝，需显式调小batch_size且不能小于2（BatchNorm需要）；不会静默产生零batch训练。全体训练记录需同时含0/1和至少两个来源。与作者相同随机组batch，不能保证每个batch都有两类，样本不足时损失可能退化；小样本调试不等于获得有效检测器。

输出每层`layer_18.pt`及同名JSON元数据，包含模型state_dict、结构、层号、来源映射、训练配置、特征配置和loss历史。已有同名模型拒绝覆盖。该checkpoint用于加载应用，不包含优化器恢复状态，不支持中断后续训。

```python
from rcs_projection import load_projection, project_vectors
model, info = load_projection('D:/CodexProjects/PrivaSee/data/outputs/rcs_projection/run01/layer_18.pt')
# X需来自相同模型、相同预处理和info['layer']层，形状[N,2048]
Z = project_vectors(model, X)  # [N,256]，原始投影输出
```

加载使用`weights_only=True`，模型切换到eval模式。输出未单位化，与作者apply_learned_projection一致；KCD单位化是后续检测器步骤。接口不自动选层、不实现MCD/KCD、不校准阈值、不输出恶意概率。

## 验证

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s backend/tests -p test_rcs_pipeline.py -v
```

CPU测试包括：网络/损失与官方AST一致、梯度有限、完整向量读写、失败后继续、重跑跳过、输入变更/损坏缓存重提取、训练标签检查、训练/加载输出一致。测试中的人工向量仅检查实现，不能证明安全检测效果。
