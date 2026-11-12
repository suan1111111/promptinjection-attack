# 图片扰动约束与目标回答优化

入口：[optimize_qwen_pixels.py](optimize_qwen_pixels.py)。默认 `--constraint global` 保持原来的全图优化流程；新增的四种形式也使用冻结的本地 `Qwen2.5-VL-3B-Instruct`，优化图片参数，不修改模型权重、原图和 CSV。当前实现是白盒，需要对图片反向传播；不是仅凭聊天 API 就能运行。

共同目标：仅在 `split=train` 上最小化 CSV 指定 `target_answer` 的自回归生成损失，平均每次选中的问题梯度。不是攻击分类标签，也不会把目标回答拼进推理时的用户问题。完整目标回答是默认模式；保留原先的 40 条训练记录、4 问题/更新、50 次更新、断点恢复、串行任务队列和留出问题生成验证。每张图片单独优化；不声称贴片可以跨图片或跨模型通用。

| 参数 | 优化对象与限制 | 能据此声称什么 |
|---|---|---|
| `--constraint border` | 只修改模型缩放后四周指定宽度的边框 | 已有 VLM 边框攻击的约束形式适配；不是 Agent Smith 原实验复现 |
| `--constraint lowfreq` | 只优化二维正交 DCT 的左上低频系数块，经逆变换生成扰动 | 低频参数化的目标回答优化；是否能操控当前模型待实测 |
| `--constraint flower` | 花形/自定义轮廓内的像素；可从真实花朵 RGBA 素材初始化 | 不规则形状贴片；自然程度与行为效果需要分别检查 |
| `--constraint sparse` | 梯度更新后保留至多 k 个 RGB 像素位置；支持集可更新 | L0 约束的目标回答优化适配；不是 Sparse-RS 复现 |

低频的像素裁剪会产生频率泄漏，8 位 PNG 量化也会引入高频。因此这里不承诺最终 PNG 严格带限，`result.json` 分别报告保存前和保存后的频谱泄漏。`frequency-ratio` 是每个轴保留比例，默认 0.125，系数面积约为原空间的 1/64。步长按逆 DCT 后的最大像素变化归一化；epsilon 约束潜在扰动幅度。

稀疏预算按像素位置计数，一个位置的 RGB 三通道合计一个像素，不是三次。先用目标损失梯度更新，再按每个位置 RGB 的 L2 范数保留 top-k；这是白盒迭代硬阈值方法。初始扰动为零，由首轮梯度选择位置。代码可运行不代表少量像素具有足够的行为操控能力；不要据此加入“已成功攻击”数据集。

花形模式有两种用途：不传素材时，只在数学花瓣轮廓内优化原图像素，不能称为“自然花朵”；传 `--flower-image` 时，将素材粘贴进该区域，再在素材附近优化。RGBA 素材以 alpha≥128 的轮廓为掩码；RGB 素材默认用花瓣轮廓裁剪。可用 `--mask-image` 覆盖轮廓（灰度白色或透明图 alpha≥128 为有效区域）。素材缩放到指定贴片宽高，轮廓外严格保留原图。epsilon 相对于粘贴后的花朵参考图，而不是原图；因此花朵出现本身可以造成远大于 epsilon 的像素变化。默认 8/255 可限制花朵颜色漂移，但不是感知自然度保证。

## 运行

在 `D:\CodexProjects\PrivaSee` 下运行，图片/CSV 路径替换成自己的文件。这里的参数是起始设置，不代表已验证最优。坐标、边框宽度、稀疏预算都针对 **Qwen 实际缩放后的输入图**。

```powershell
# 边框。先预检模型输入几何，不加载模型权重。
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --image INPUT.png --csv TASKS.csv --constraint border --border-width 16 --validate-only
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --image INPUT.png --csv TASKS.csv --constraint border --border-width 16 --steps 50

# 低频参数化
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --image INPUT.png --csv TASKS.csv --constraint lowfreq --frequency-ratio 0.125 --steps 50

# 真正花朵素材 + 不规则轮廓。优先使用背景透明的 PNG。
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --image INPUT.png --csv TASKS.csv --constraint flower --flower-image FLOWER.png --patch-width 84 --patch-height 84 --patch-x 0 --patch-y 0 --epsilon 8 --steps 50

# 稀疏像素。默认 256 个 RGB 位置；可独立考察更大的幅度预算。
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --image INPUT.png --csv TASKS.csv --constraint sparse --sparse-pixels 256 --epsilon 32 --steps 50
```

非全图模式允许 epsilon 最高 255/255，以支持边框/稀疏/贴片的幅度对照；alpha 不得大于 epsilon。增大花朵 epsilon 会削弱其参考外观限制。原全图模式仍保持原先最高 64/255 的限制。

恢复时给出相同输入与约束参数，用 `--resume 输出目录/checkpoint.pt --steps 新的总次数`。检查点中的 `delta` 在低频模式下是 DCT 系数，不是图片像素差；不要直接把它当图片。PNG 为导出结果，恢复优化以 checkpoint 为准。模式、掩码/素材哈希、位置、预算等变化会被拒绝。只加载可信的本地 checkpoint。

队列 JSON 支持原来的字段及上述约束参数，键名用下划线，例如 `constraint`, `border_width`, `flower_image`, `sparse_pixels`。素材路径和 image/csv 一样相对于 JSON。每个任务必须有 `image` 和 `csv`。所有任务串行运行，完成跳过，失败可以重启恢复。

## 验证与输出

- `clean_resized.png`：原图经模型缩放后的基线。
- `constraint_reference.png`：约束参考图，花朵模式为粘贴素材后、优化前的图。
- `constraint_mask.png`：边框/花形有效区域；低频/稀疏模式此图为全白，不表示实际支持位置。
- `adversarial_initial.png` / `adversarial_final.png`：随机初始化或零初始化后的输入，以及最终导出 PNG。
- `perturbation_visualized.png`：相对原图的放大差异，花朵模式包含花朵粘贴效果，不能用于测量素材内扰动预算。
- `training_log.csv` / `checkpoint.pt` / `config.json`：训练日志、可恢复参数、输入和设置证据。
- `result.json`：实际变化像素数、相对原图/参考图的变化幅度、掩码外变化或低频泄漏。成功状态保持 `unassessed`。
- `evaluation.json`：重新读取 8 位 PNG，对 `split=test` 问题生成原图与扰动图回答。花朵模式额外生成未优化参考图回答，区分“放了一朵花”的影响与像素优化的影响。

CSV 使用原有字段 `split,category,question,normal_answer,target_answer`，任意新类别请保持 `--target-mode full`。可选 `target_contains` 只提供字面标记匹配。正常任务是否保留、偏移是否符合攻击目标、是否属于提示注入，需要结合输出独立判断；loss 下降、单步梯度通过、生成完成、出现字面标记均不自动等于语义攻击成功。`--eval-rows 0` 不生成回答；`--gradient-check-only` 只做单步链路检查，不验证行为。

真实数据优先纳入有原图、原任务和攻击目标且在当前模型上验证成功的样本；其余保留为尝试/失败/边界样本。低频、花形和稀疏适配未验证成功前，不应作为三个已成立的新型提示注入攻击来计数。

工程检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s pixel_attack_trials -p test_pixel_constraints.py -v
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --self-test
# 可选：生成无害合成图与标记目标，四个任务各跑一次真实梯度和留出问题生成。
.\.venv\Scripts\python.exe pixel_attack_trials/make_constraint_smoke_inputs.py
.\.venv\Scripts\python.exe pixel_attack_trials/optimize_qwen_pixels.py --tasks pixel_attack_trials/constraint_smoke_inputs/tasks.json --output-dir pixel_attack_trials/constraint_smoke_outputs --eval-max-new-tokens 32 --save-every 1
```

## 来源与改造边界

- [Agent Smith 官方代码](https://github.com/sail-sg/Agent-Smith)：已有面向 VLM 的边框攻击入口，原实验使用 LLaVA。这里复用本项目 Qwen 的目标回答优化框架实现边框掩码，不复制其多智能体传播实验、数据和增强设置。
- [Low Frequency Adversarial Perturbation](https://arxiv.org/abs/1809.08758) / [官方代码](https://github.com/cg563/low-frequency-adversarial)：借鉴限制到低频空间的思路，改成可微 DCT 系数的生成式回答目标；不是原论文黑盒分类算法的复现，也不引用其分类成功率作为本程序结果。
- [Sparse-RS](https://github.com/fra31/sparse-rs)：说明已有稀疏/边框等扰动约束。这里独立写白盒 L0 硬阈值适配，不将其分类器攻击结论迁移为 VLM 注入结论。
- [TnT](https://tntattacks.github.io/)：提供花朵等自然外观分类攻击的动机。这里支持花朵参考图和不规则掩码，不实现其生成器，不声称实现其自然外观保证或 VLM 指令操控结果。
