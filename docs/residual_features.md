# 残差数值提取

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe backend\extract_residual.py "图片路径.png"
```

直接打印 6 个数值，无需 OCR、模型推理、GPU、MATLAB 或新依赖。`--store 目录` 可指定缓存；`--groups kb` 可只计算 3×3 部分。

## 原始代码依据

SRM 作者 DDE Lab 的[官网源码下载页](https://dde.binghamton.edu/download/feature_extractors/)提供 [SRM.zip](https://dde.binghamton.edu/download/feature_extractors/download/SRM.zip)，其中包含 SRM.m。下载文件的 SHA256 与原始许可见 [SRM_LICENSE.txt](references/SRM_LICENSE.txt)。

本实现移植 SRM.m 中 `Residual(X,2,'KB')` 和 `Residual(X,3,'KV')` 的线性残差公式，分别对应 3×3 与 5×5 滤波器。KB 排除一圈边界，KV 排除两圈边界，不填充、不缩放、不量化、不改变作者系数。数学公式已用独立逐点实现验证，未运行 MATLAB 对照。

截至本次检索，没有确认 CVPR 2019《Detection Based Defense Against Adversarial Examples From the Steganalysis Point of View》的作者独立源码仓库。该文采用 SPAM/SRM 特征，本项目依据的是 SRM 原作者代码，不是该文完整防御代码。

## 输出定义与范围

每个滤波器分别输出：

| 后缀 | 计算 |
| --- | --- |
| residual_mean_abs | 残差绝对值均值 |
| residual_std | 有符号残差的总体标准差，ddof=0 |
| residual_energy | 残差平方均值 |

前缀 `kb_`、`kv_` 区分滤波器。先按 EXIF 校正方向、透明背景合成白色，再使用 Pillow L 灰度转换，像素单位为 0～255。输出不是 0～1；残差幅值可能超过 255。不同滤波器的系数尺度不同，其结果不能直接当成同一个阈值比较。

这 6 个汇总值是项目增加的轻量统计，不是原论文的 34671 维 SRM 特征，不包含共生统计、FLD 分类器、修改概率估计或攻击判断。滤波保留的正常边缘、纹理及压缩痕迹也影响残差；均匀亮度变化会被这些零和滤波器消除。结果不等于攻击概率，也不保证覆盖所有全局扰动。

## 增量缓存

复用 `data/image_features/features.sqlite3`，以图像文件内容 SHA256、配置版本、滤波器分组为键。KB、KV 各自提交事务，第二组失败时第一组保留；重跑只补未完成组。原图另存缓存 `original.bin`，不导出 JSON，不保存大体积残差图。频率、梯度、熵、OCR 的缓存使用不同配置与分组。

至少 5×5 的图片才能完成两组。3×3 或 4×4 图片可以计算 KB，KV 会明确失败，不会保存伪造的零值。
