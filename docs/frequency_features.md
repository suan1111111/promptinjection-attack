# 图片频域数值提取

在项目根目录执行（PowerShell）：

```powershell
.\.venv\Scripts\python.exe backend/extract_frequency.py picture/medical-page-09.png
```

终端仅输出 6 项 DE-FIVE 分量：RGB 每通道各 1 项高低频功率比、1 项熵替代项。每行是 `字段名: 数值`，不导出 JSON 文件，不计算综合分数，也不计算或新增缓存旧版的 45 项统计。

DE-FIVE 分量逐通道保留高低频功率比与熵替代项：低频半径为 8 个 FFT 频点，高低频比包含 DC，熵项为幅值分布的 `sum(q * log(q))`（通常为负数），与下表的归一化功率谱熵不同。沿用此前实现，不混合两项、不平均 RGB 通道。原单分数接口及缓存保留，当前终端入口不调用它。

DE-FIVE 直接复用或计算功率谱，不依赖旧版统计。终端入口已移除 `--groups` 参数。已有旧版缓存不删除，服务的 `extract()` / `extract_bytes()` 与原 Flask 接口仍保留旧版统计能力；下文旧版字段表仅描述这些接口。
缓存目录为 `data/image_features/<文件SHA256>/<配置ID>/`。
SQLite 保存各组记录，`power.npy` 保存可复用功率谱，`original.bin` 保留原始文件字节。
后端使用同一服务时调用 `FrequencyFeatureStore.extract()` 或 `extract_bytes()`。
已有 Flask 服务注册 `POST /api/image-features/frequency`：multipart 字段 `file`，可选 `groups=bands,statistics`。

## 数值定义

原图不缩放、不压缩。EXIF 方向校正；透明通道合成白背景；解码成 RGB8 并除以 255。
高位深、非 RGB 色彩模式会经 Pillow 转成 RGB8，此转换写在配置中；不进行 ICC 色彩管理。
每个 RGB 通道独立计算二维正交归一化 FFT，无额外加窗。功率为幅值平方。
频率坐标用 cycles/pixel，径向距离除以二维理论最大半径 sqrt(0.5²+0.5²)，范围 0–1。
DC 指 (0,0) 的平均亮度成分，频带占比和分布统计排除 DC。

| 字段 | 定义 |
|---|---|
| ac_energy | 非 DC 功率总和 |
| low/mid/high_energy | 半径 [0,0.15)、[0.15,0.5)、[0.5,1] 的非 DC 功率 |
| low/mid/high_ratio | 对应能量除以 ac_energy |
| total_energy | 含 DC 功率；正交 FFT 下等于归一化像素平方和 |
| dc_energy_ratio | DC 功率占全部功率的比例 |
| centroid | 非 DC 功率加权平均半径 |
| bandwidth | 半径相对于 centroid 的功率加权标准差 |
| entropy_normalized | 非 DC 频点功率分布的 Shannon 熵除以 log(频点数量) |
| peak_energy_ratio | 单个最强非 DC 频点占 AC 能量的比例（不是缩放攻击检测） |
| radius_q50/q90 | 累积非 DC 能量达到 50%/90% 的半径 |

没有 AC 信号时占比和分布统计为 null；未计算或失败的组也用 null 并标明状态。
仅排除机器舍入量级的 AC 能量（相对总能量 1e-24，绝对 1e-30）。
频带边界是本系统的初始实验配置，不是论文验证的攻击阈值。
周期边界和无加窗可能产生频谱泄漏，因此必须比较相同配置下的数据。

## 持久化与复用

每完成一组单独提交 SQLite。失败组可以重试，成功组不会重算。
同一字节内容、同一配置共用结果；文件名不参与缓存身份。算法/参数/依赖版本参与配置 ID。
重复或并发请求通过 SQLite 写事务避免重复计算。原图、频谱和数值均保留。
服务被强制终止后已提交数据仍在，重新执行会补齐缺失项。
这版是单机项目缓存，尚未实现用户权限隔离；接入多用户流程时需增加所属用户检查。
不对动画或多帧图片自动选择帧，明确拒绝此类输入。

## 研究依据和边界

参考 [SpectralDefense](https://arxiv.org/pdf/2103.03000) 的输入 RGB 傅里叶分析。
论文使用完整幅值频谱训练分类器；这里提取紧凑功率统计，不是完整复现，尚不输出攻击判断。
特定缩放峰检测需要目标缩放流程，不包含在当前实现中。
依赖复用项目已有 NumPy、Pillow、Flask，无须新模型或 GPU。

验证：`.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -p test_frequency_features.py -v`
