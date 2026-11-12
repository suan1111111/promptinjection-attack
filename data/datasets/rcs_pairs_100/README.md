# RCS 100 对数据集（200 张）

用于隐藏状态检测（RCS）的配对数据集：每个 `pair_id` 包含一张攻击图和一张对应的良性/干净图，同尺寸、同背景、只差注入内容。

## 结构

```
rcs_pairs_100/
├── pairs.json                 # 100 条，主清单
├── index.csv                  # 同样的内容，表格版
├── text/                      # 文字注入，50 对
│   ├── visible_command/       #  5 对
│   ├── low_contrast/          #  5 对
│   ├── typographic_obfuscation/  5 对
│   ├── small_corner_text/     #  5 对
│   ├── cross_region_completion/  5 对
│   ├── reachable_qr/          #  5 对
│   ├── quotation_ambiguity/   #  5 对
│   ├── authority_impersonation/  5 对
│   ├── task_relevant_disguise/   5 对
│   └── reachable_metadata/    #  5 对
└── pixel/                     # 像素扰动，50 对
    ├── region/                # 10 对
    ├── global/                # 10 对
    ├── border/                # 10 对
    ├── fixed/                 # 10 对
    └── flower/                # 10 对
```

文件命名：

- 文字侧：`<pair_id>_a.png`（攻击）/ `<pair_id>_b.png`（良性）
- 像素侧：`<kind>_<原图id>_adv.png`（攻击）/ `<kind>_<原图id>_clean.png`（干净）

## pairs.json 字段

| 字段 | 说明 |
|---|---|
| `pair_id` | 唯一 ID，如 `text/visible_command-0004-pair`、`pixel/flower/coco_8` |
| `domain` | `text` 或 `pixel` |
| `category` | 文字侧为 10 个家族之一；像素侧为 5 种扰动之一 |
| `attack` / `clean` | 相对本目录的路径 |
| `attack_text` / `clean_text` | 文字侧图上渲染的文字；像素侧为 null |
| `group` | 分组键：文字侧 `coco:<背景id>`，像素侧 `orig:<原图id>`，用于按组切分 |
| `background_id` | 文字侧的 COCO 背景 id；像素侧为 null |
| `split` | 文字侧的原始 split；像素侧为 null |
| `source` | 原始文件路径（相对项目根目录） |

## 选择规则

- 文字侧：10 个家族各 5 对，按 `pair_id` 排序后等间隔抽取，保证同一家族内分散；50 对来自 50 个不同 COCO 背景。
- 像素侧：5 种扰动各 10 对，取 `status.json` 中 `state=effective` 的全部结果；12 张不同原图。

## 分组与切分

- 独立分组共 **62 个**：文字侧 50 个背景 + 像素侧 12 张原图。
- 训练/验证/测试必须按 `group` 切，不能按图片切：像素侧同一原图有 5 种扰动，文字侧要防止同背景跨集。
- 建议：按 group 做 70/15/15 或留出 10 个 group 做测试。

## 已知待办

1. **问题还没配**：RCS 特征提取需要 `(图片, 问题)`。文字侧图上文字在 `attack_text`/`clean_text` 里；像素侧需要从各自 `datasets/<id>.csv` 的 train 前 40 条里选。
2. 文字侧良性图是“同背景 + 良性文字”，不是无文字原图；这是刻意的控制变量设计。
3. 像素侧图片是模型输入尺寸（如 364×252），文字侧是 COCO 原始尺寸；提取时会统一由 processor 缩放。

## 来源

- 文字侧：RI Multimodal Injection（`hanIshi/ri-multimodal-injection`，图像背景为 MS-COCO 2017）。
- 像素侧：本项目 `outputs/scene_*_top10_*` 五个批次的 effective 结果。
