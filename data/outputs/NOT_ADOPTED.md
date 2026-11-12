# 未采用的实验（2026-11-11）

部署版本仍是 `rcs_artifact_v2`（层 9，margin 1.0，阈值 −0.05）。以下实验已取消，不再使用：

| 路径 | 内容 | 取消原因 |
|---|---|---|
| `data/outputs/rcs_projection_v2_benign/` | 加 500 张干净 COCO 背景（未配平） | 文字 val 误报升到 77%，模型学成“有字即可疑” |
| `data/outputs/rcs_projection_v2_balanced/` | 良性降采样到 505:505 | 像素 AUROC 0.62，v2 val AUROC 0.75，比旧版更差 |
| `data/outputs/rcs_features_balanced/` | 配平后的特征缓存 | 同上，仅服务于配平实验 |
| `data/outputs/rcs_features_combined/` | v2 + 良性池合并缓存 | 仅服务于良性池实验 |
| `data/outputs/rcs_features_benign/`、`rcs_features_benign_all/` | 干净 COCO 背景特征 | 仅服务于良性池实验 |
| `data/datasets/benign_coco/` | 500 张干净 COCO 背景图 | 仅服务于良性池实验 |

保留这些文件是为了可追溯；不要把它们当作部署输入。

## 补充：合并缓存的 split 泄漏（2026-11-11 修复）

初版 `build_benign_coco.py` 按图片 ID 哈希重新分配 train/val，没有沿用 v2 原始 split，导致：

- `rcs_features_combined`：170 个原图分组同时出现在不同 split
- `rcs_features_balanced`：140 个
- `rcs_features_v2`：0（默认产物不受影响）

这两个合并缓存上的评估结果因此被污染，只能当作粗略参考。修复后 `build_benign_coco.py` 直接沿用 v2 中对应背景的 split；用修复后的清单重建 `rcs_features_combined_fixed`，跨 split 分组为 0。

另外三个“贴近论文”的实验也不采用（仅留档）：

| 路径 | 结果 |
|---|---|
| `data/outputs/rcs_projection_v2_paper/` | 论文原样（含紧凑项、margin 2.0）：完全坍缩，像素 AUROC 0.45 |
| `data/outputs/rcs_projection_v2_eq3/` | 去紧凑项、margin 2.0：AUROC 0.85，但阈值饱和、全召回误报 0.60 |
| `data/outputs/rcs_projection_v2_eq3_m15/` | margin 1.5：AUROC 0.65 |
