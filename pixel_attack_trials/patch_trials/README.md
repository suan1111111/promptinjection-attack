# Qwen 固定与随机位置贴片实验

复用上层 `optimize_qwen_pixels.py` 的离线配置、Qwen 可微预处理、答案掩码和图片保存。
模型为现有 Qwen2.5-VL-3B-Instruct NF4；不依赖下载的 Image Hijacks 软件环境。

数据划分严格隔离：只有 `split=train` 参与梯度更新，`split=val` 只用于选择检查点，
`split=test` 只在权重已经落盘之后评估。最终结论以保存后的 PNG 和实际生成回答为准，
loss 或探针损失只说明优化在动，不代表注入成功。

## 运行

以下 PowerShell 命令均从 `D:\CodexProjects\PrivaSee` 执行。下面的命令用于初步试跑
（100 步、56×56 贴片、每步4个问题各配一个随机位置、SGD lr 3、无动量、验证集选点、9 个测试回答）。
该小规模结果仅供探索，不能当作最终成功率，也不能单凭未命中排除该方法。

随机贴片与固定贴片统一使用56×56。在364×252的模型输入上占约3.42%。
此前216×216的实验保留为历史记录；已训练的大贴片不能通过缩小图片替代重新优化。

```powershell
.\.venv\Scripts\python.exe .\pixel_attack_trials\patch_trials\train_qwen_patch.py `
 --mode random --init random --position-sampling full `
 --image D:/CodexProjects/PrivaSee/data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/clean_images/0.png `
 --csv D:/CodexProjects/PrivaSee/data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/instruction_data/0/Attack/dataset.csv `
 --target-mode full --max-train-rows 40 --max-visual-tokens 128 `
 --patch-width 56 --patch-height 56 `
 --optimizer sgd --lr 3 --momentum 0 `
 --questions-per-step 4 --positions-per-step 4 `
 --steps 100 --save-every 25 --seed 7 --probe-every 50 --probe-positions 4 `
 --val-every 50 --val-positions 2 `
 --eval-rows 3 --eval-positions 3 --eval-max-new-tokens 512
```

```powershell
# 仅核对输入、划分、目标标签、处理器和贴片几何；不加载模型权重
... 同样的参数 ... --validate-only

# 单步反向传播和更新检查（会加载模型，保存诊断图片）
... 同样的参数 ... --gradient-check-only
```

默认：56×56、前5条训练记录、50次更新、每10步保存。
尺寸和坐标均针对模型预处理后的图片（由输入和视觉token预算决定），不是原始PNG。
固定模式下贴片固定在默认位置，并从原图该区域初始化，用符号梯度（步长1/255）；
随机模式每步平均 `--positions-per-step` 个随机位置（默认4个），
`--questions-per-step` 个问题按轮转分配（4问+4位置=每问一个位置，共4次前向）。
模型和背景冻结；只更新贴片，像素裁剪到[0,1]。这是覆盖式贴片，没有原图epsilon限制。

## 随机位置训练的四个要点

- 单个随机位置只给出期望损失的一个样本。**符号梯度**会把这个高方差估计放大成随机游走：
  同一初始贴片、同样12步，符号梯度（±2/255）让探针损失升高0.017，而SGD（lr 3）降0.052、
  Adam（lr 0.02）降0.076。因此 `--optimizer auto`（默认）在随机模式下用Adam，在固定模式下用符号梯度；
  要对齐论文的SGD，显式写 `--optimizer sgd --lr 3 --momentum 0`。
- `--position-sampling full`（默认）在整个可放置范围内均匀采样，对应官方做法；
  `content` 是可选的启发式（用逐行/列标准差跳过纯色区），它也会跳过真实的均匀图像区域，不建议默认使用。
- `probe_loss` 是**固定问题 × 固定位置**的平均损失，报告中会额外写一行 `step=0` 的
  更新前基线；它只反映损失变化，不反映是否注入成功。
- `validation_trace.json` 记录每 `--val-every` 步在10条验证问题上的损失，
  选损失最低的检查点作为最终贴片；测试集只评估这个选定结果。CSV 没有 val 划分时自动退回“用最后一步”。
- **生成网格必须与训练网格一致**。处理器会根据输入图片重新算网格，`min_pixels` 甚至会把
  已经是训练尺寸的图放大（例如 252×168 → 280×196，54 → 70 个视觉 token），这样损失降到 0
  贴片也毫无效果。脚本启动时会校验生成网格 == 训练网格，不通过直接报错；生成一律使用
  `min_pixels=4·28·28` 的处理器。评估用的 `adversarial_final.png` 必须按训练分辨率喂给模型，
  贴回原始大图会重采样导致失效。

## 参数

- `--x 100 --y 20`：指定固定模式左上角；随机模式中只影响预览和crop初始化位置。
- `--patch-width 56 --patch-height 56`：贴片尺寸，与固定贴片一致；启动日志会打印它占模型输入的面积比例和视觉token数。
- `--positions-per-step 1`：每次更新只看一个随机位置（方差最大，不建议）。
- `--questions-per-step 4 --positions-per-step 4`：每步4个不同问题、各配一个独立随机位置，梯度平均后更新一次。
- `--optimizer sign --alpha 4`：改用符号梯度并放大步长；`--momentum 0.9` 只对 sign/sgd 生效。
- `--probe-every 50 --probe-positions 4 --probe-questions 4`：探针频率、位置数、固定问题数（`--probe-every 0` 关闭）。
- `--val-every 100 --val-positions 2`：检查点选择频率与每个验证问题的固定位置数（`--val-every 0` 关闭）。
- `--gen-check-every 100 --gen-check-questions 2 --gen-check-max-tokens 48`：每 N 步用当前贴片真生成一次，
  回答是否出现目标句直接写进 `generation_checks.json`，用来提前判断该不该继续跑（默认 0 关闭）。
- `--init random`：固定和随机实验采用相同随机初始化，便于对照。
- `--max-train-rows 0`：使用所有train行；`--steps 1000`：1000次更新，不是1000个epoch。
- `--target-mode full`：使用CSV的target_answer；short优先用short_target列，否则沿用类别短目标。
- `--clip-grad 20`：sgd/adam的梯度L2上限（`0` 关闭）；该设置会写进 `config.json`。
- `--eval-rows 10 --eval-positions 10`：10个测试问题 × 10个位置 = 100个回答，
  外加每个问题的原图回答；位置和判定结果都写进 `evaluation_positions.json`。
- `--seed 7`：记录并控制初始化、问题顺序和位置采样，CUDA运行不保证逐位确定性。

## 输出

`patch_trials/outputs/random_时间戳/`：

- `config.json`：参数、输入哈希、实际尺寸、解析后的优化器/学习率、位置范围、探针与评估位置。
- `training_rows.json`：train / val / test 三组实际使用的问题、目标和CSV行号。
- `clean_resized.png`：模型实际使用的干净背景。
- `patch_initial.png`、`initial_preview.png`：初始贴片与预览图。
- `training_log.csv`：每步问题行号、实际坐标（列表）、更新前loss、梯度、区域外差值、显存、`probe_loss`、`val_loss`；第一行是 `step=0` 的基线。
- `patch_step_*.png`、`full_step_*.png`：该步贴片和该步最后一个位置的合成图。
- `patch_final.png`：**验证集选中的**贴片；`patch_last_step.png`：最后一步的贴片。
- `full_final_preview.png` / `adversarial_final.png`：选中贴片在预览坐标的合成图。
- `patch_placements_grid.png`：选中贴片贴在探针位置上的拼图。
- `validation_trace.json`：验证损失轨迹与选中步数。
- `evaluation_positions.json`：测试集多位置回答与判定汇总。
- `patch_final.pt`：选中贴片（在生成评估**之前**写入，生成失败也不会丢权重）。
- `result.json`：选中步数、基线探针/验证损失、评估汇总和运行检查。

`evaluation_positions.json` 里每条回答记录：完整目标句是否出现（`target_full_match`）、
正常回答是否保留（`normal_match`）、CSV标记是否出现（`marker_match`），以及分类
`target_only / target_and_normal / normal_only / other / degenerate`；每个问题另存
原图回答及“原图回答是否已包含目标句”。这些是字面匹配，语义是否真的注入仍需人工看回答；
只出现 `<maliciousurl>` 不构成完整目标满足。

PNG采用8-bit量化，训练在浮点像素上进行；重新加载PNG后的实际生成效果需另行评估。
CUDA日志是PyTorch分配内存，不等于整块显卡的所有进程占用。
原图、CSV、已有脚本和已有outputs均不覆盖。

## CPU实现检查

```powershell
.\.venv\Scripts\python.exe .\pixel_attack_trials\patch_trials\test_patch.py
```

测试贴片区域梯度、边界位置、背景不变及train/test/val隔离。
