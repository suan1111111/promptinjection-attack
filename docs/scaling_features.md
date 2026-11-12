# 缩放攻击数值特征

来源：[作者源码](https://github.com/EQuiw/2023-detection-scalingattacks)与
[On the Detection of Image-Scaling Attacks in Machine Learning (ACSAC 2023)](https://arxiv.org/html/2310.15085v1)。
本实现提取其中 Peak Spectrum 和 Down/Up Scaling (PSNR) 两项，并非复现论文全部检测器。

```powershell
.\.venv\Scripts\python.exe backend\extract_scaling.py "图片路径.png"
# 指定下游缩放配置，尺寸顺序是高、宽
.\.venv\Scripts\python.exe backend\extract_scaling.py "图片路径.png" --target 224 224 --algorithm nearest
```

终端输出：

| 字段 | 含义 | 方向 |
| --- | --- | --- |
| peak_spectrum_percentile | 预测峰窗口的平均对数频谱强度在整幅频谱中的百分位，0～100 | 高值可能可疑 |
| down_up_psnr_db | 按指定算法缩小，再放大到原尺寸，与原图比较的 PSNR，单位 dB | 低值可能可疑 |

默认 Pillow 最近邻、224×224，采样半径 5。尺寸与算法必须对应实际下游预处理；默认配置不代表 PrivaSee/Qwen 的实际配置。其他选项 bilinear、bicubic、lanczos 使用 Pillow 实现，同名算法在 OpenCV 或其他库中不一定等价。

`f_utils.py` 直接复制作者文件，保留 GPL-3.0 许可证与来源。频谱使用 RGB 平均值、FFT、作者的对数变换；预测主峰、排除中心区、采样窗口均保持原代码。半径 5 的窗口按作者切片规则宽 10，而非 11。百分位遵循作者 SciPy 默认 rank 规则。PSNR 用数学公式替代作者已弃用的 skimage 接口，RGB8 数据范围为 255。

本项目新增输入解码（EXIF 方向、透明背景合成白色）、CLI 和持久化缓存。无预测峰时显示 `NA`，不把作者空窗口产生的 NaN 当成有效数值；完全无重建误差时 PSNR 显示 `inf`。目标必须至少缩小一个轴，不能放大任何轴。只支持单帧图像。

缓存沿用 `data/image_features/features.sqlite3`，按图像内容哈希、缩放配置与算法版本隔离；两项各完成一项即保存一项。某项失败时终端报告错误并返回非零退出码，下次重试复用已完成项。无需用户导出 JSON，也无需 OCR 或模型推理；已有频率、梯度、熵和 OCR 特征不受修改。

这些是异常特征，不是攻击概率，没有通用判断阈值。自然纹理也可能产生高频峰或较低 PSNR，适应性攻击可绕过频谱方法。本次验证包括数值计算、人工构造的缩放隐藏图像和缓存恢复，尚未用真实提示注入数据集评估检出率；也不判断缩放后内容是否具有恶意语义。
