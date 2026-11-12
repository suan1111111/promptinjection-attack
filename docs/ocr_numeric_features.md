# OCR 数值特征

在项目根目录运行（首次自动识别需要 OCR 后端已启动）：

```powershell
.\.venv\Scripts\python.exe backend\extract_ocr_features.py "图片路径.png"
```

已有 DeepSeek 带 `<|ref|>` / `<|det|>` 标记的定位结果可用 `--ocr-input result.txt`；已有 Tesseract 词级 TSV 可用 `--ocr-input result.tsv --format tsv`。TSV 坐标必须对应经过 EXIF 方向校正的原图，不能使用缩放图坐标。不会安装或调用 Tesseract。

只在终端打印以下 16 项，不导出 JSON 文件、不添加界面：

| 数值 | 定义 |
| --- | --- |
| character_count | OCR 非空白字符数 |
| text_box_count | 文字框数量 |
| confidence_available_count | 有置信度的框数量 |
| confidence_mean / confidence_min | 可用框置信度均值 / 最小值，0～1 |
| text_area_ratio | 文字框并集面积 / 图像面积 |
| relative_box_height_mean / relative_box_height_max | 框高度 / 图像高度的均值 / 最大值 |
| edge_text_box_ratio | 接触图像外围 5% 区域的文字框比例 |
| box_background_contrast_mean | 各框内部像素相对框外 3 像素环带灰度中位数的最大差值，再取框间均值，0～1 |
| gradient_text_coverage / entropy_text_coverage | 重合面积 / 文字框并集面积 |
| gradient_response_in_text_ratio / entropy_response_in_text_ratio | 重合面积 / 对应响应区域面积 |
| gradient_iou / entropy_iou | 重合面积 / 两区域并集面积 |

空分母或缺失统计输出 `NA`。DeepSeek 当前定位输出没有置信度，不能用它推算识别准确率；其框通常是版面块，框高度不等于字体大小。灰度对比度只是框与周围背景的代理统计，不是精确文字前景对比度。框内插图、纹理也可能影响它。

首次需要结构化 OCR 才能知道位置：仅有之前的纯文字缓存不够，会通过现有后端请求 markdown grounding OCR。后续按图片内容复用结构化 OCR。梯度区域复用已有选中窗口的并集；熵区域使用已有 Jedi 实现过滤后的候选掩码。这些是统计特征，不是攻击概率。

默认保存到现有 `data/image_features` SQLite 缓存，可通过 `--store` 指定目录。结构化 OCR、质量、几何、梯度重合、熵重合分别提交；后续失败不会撤销已完成组，重新运行只补未完成组。更换 OCR 输入会更新其布局缓存并按新输入生成统计。内部 SQLite 存储使用 JSON 字段，不产生 JSON 导出文件。

测试覆盖已知几何、重合比例、缺失置信度、坐标检查及中断后恢复。自动 OCR 后端的实际识别效果仍需用真实图片验证。
