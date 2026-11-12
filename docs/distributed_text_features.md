# 跨区域文字组合检测：独立程序

```powershell
.\.venv\Scripts\python.exe backend\extract_distributed_text.py "图片路径.png"
# 复用已有定位 OCR 的原始输出，不调用模型
.\.venv\Scripts\python.exe backend\extract_distributed_text.py "图片路径.png" --ocr-input result.txt
# 复用已有 Tesseract 词级 TSV，不安装或调用 Tesseract
.\.venv\Scripts\python.exe backend\extract_distributed_text.py "图片路径.png" --ocr-input result.tsv --format tsv
```

只打印下列 5 项结构特征；不再打印五类单独次数或涉及框比例。内部仍保留类别与片段证据。

| 字段 | 计算 |
| --- | --- |
| cross_box_match_count | 去重后的跨框新增类别命中总数；同一片段命中两类计两次 |
| mean_combination_box_count | 各命中最短片段包含的框数均值 |
| max_combination_box_count | 各命中最短片段包含的框数最大值 |
| mean_box_gap_normalized | 命中片段中相邻矩形框的最短欧氏距离，除以原图对角线，再对所有相邻框对取均值 |
| max_spatial_span_normalized | 每个命中片段内最远两个框中心的距离，除以原图对角线，再取所有片段最大值 |

结构统计先合并完全相同的框索引序列，避免一个片段因命中多个类别而重复加权。重叠但不同的片段各自保留，框间空隙统计中各片段内的每个相邻框对权重相同。矩形重叠或接触时空隙为 0。两项归一化距离范围为 0～1；框数不归一化。没有新增命中时总数为 0，其余四项显示 `NA`。这些是项目设计的描述特征，不是论文定义的攻击分数，也不是最终攻击判定。

## 论文依据与实现边界

[XYLayoutLM，CVPR 2022，第 3.3 节及附录](https://arxiv.org/html/2203.06947v1#S3.SS3) 提供投影、空白谷、交替 X/Y 切分和无法切分时的 Y/X 排序。原文明确：训练阶段使用随机增强，**推理阶段只用普通 XY Cut**。本模块按这些定义独立实现确定性推理排序：从 Y 轴开始、闭整数区间、零投影谷切分、交替轴、无法双轴切分时按上到下再左到右排列。用区间并集替代整张投影数组；相同坐标按输入索引稳定排序。没有运行 XYLayoutLM 神经网络，也没有复制第三方 Rust/Python 仓库的源码，不宣称作者源码逐字复现。

跨区域攻击依据：[IPI 第 III-D 节](https://arxiv.org/html/2603.03637v1#S3.SS4) 将长指令拆到多个区域，保持空间顺序。XYLayoutLM 是文档阅读顺序方法，不是攻击检测论文。

本项目扩展：按排序后的连续框检查长度 2～4 的窗口（`--max-blocks` 可设 2～32）；将换行归一为空格后复用现有 SnapGuard 五类规则。如果某类别已在窗口内任一单框命中，则不计新增跨框命中；包含更短已命中窗口的同类别窗口不重复计数。保存最短候选窗口及涉及框索引作为内部证据，不把整页任意框做全组合。

这些窗口上限、去重和数值定义不是论文参数，而是项目的轻量实现选择。它们不能判断跨框文字是否语义相关；无关但相邻段落也可能误命中。普通合法的“enter password”同样会命中，保留原规则含义。单个布局块内部的跨行匹配不算跨框；词级 TSV 的“框”是词，不一定是独立区域。超出窗口上限、非连续碎片、非标准阅读方向、复杂多栏/表格/箭头连接的指令不保证检出。

英文规则不支持纯中文，同义改写和 OCR 漏字也可能漏检。无命中不表示安全。首次定位 OCR 的耗时与识别质量仍取决于现有 OCR 后端；已经有原始定位文本时没有额外模型推理。

## 独立缓存与输入

使用单独的 `data/image_features/distributed_text.sqlite3`，不修改其他入口、服务或已有数据库。原始 OCR、逐框文字、排序、特征分别提交；中途失败保留此前完成阶段。原始文本、顺序及内部证据保存在 SQLite，不导出 JSON 文件。图像内容、OCR 原始输入、解析/排序版本、窗口上限和规则文件哈希隔离缓存。五项结构统计使用独立 v2 特征版本，不复用旧七项结果；旧记录保留，已有 OCR 与排序缓存继续复用。

已有 `ocr_numeric_features` 布局缓存保存了汇总文字和框，但未保留可靠的逐框文字关联，不能盲目拼配。本程序首次需要原始 grounding/TSV，或通过现有后端进行一次定位 OCR；以后复用本程序缓存。多框共用一段文字的 grounding 会报歧义错误，不重复复制文字伪造候选。TSV 像素坐标必须对应 EXIF 校正后的原图；grounding 坐标使用 0～999。仅支持单帧图片。

运行测试：`.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -p test_distributed_text_features.py -v`。
覆盖排序、分词/跨框命中、去重、单框排除、普通文本、输入关联及错误、中断恢复、已知距离、尺度归一化、多类别结构去重、旧缓存隔离和 CLI 的 NA 输出。未在真实跨区域攻击数据集上测检出率。
