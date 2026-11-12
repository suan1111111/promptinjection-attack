# 场景内扰动区域候选

本批 22 张图片仅选区域，未训练。绿色只是预览标注，不会添加到训练图片。

收音机沿用已确认范围。其他区域是人工候选，不是自动精确分割；面积不强行统一，不能据此保证效果。

masks_native 对应原图；masks_model128 按现有处理器 64–128 视觉 token 设置对齐。更改视觉预算后需要重新生成模型尺寸掩码。

后续沿用现有自定义 mask-image 接口即可，掩码白色可改，黑色保持原图；本目录不提供自动训练队列。

| 图片 | 选择区域 | 模型尺寸下占比 | 预览 |
|---|---|---:|---|
| 0.png | 收音机本体 | 34.2% | [查看](previews/0.jpg) |
| 1.png | 链锯外壳与导板 | 12.1% | [查看](previews/1.jpg) |
| 2.png | 教堂两侧塔楼墙面 | 29.0% | [查看](previews/2.jpg) |
| 3.png | 狗的躯干与腿部 | 7.6% | [查看](previews/3.jpg) |
| 4.png | 圆号金属表面 | 13.5% | [查看](previews/4.jpg) |
| coco_1.jpg | 击球者球衣 | 8.4% | [查看](previews/coco_1.jpg) |
| coco_2.jpg | 前景斑马躯干条纹区域 | 26.2% | [查看](previews/coco_2.jpg) |
| coco_3.jpg | 中间两块蓝色冲浪板 | 12.4% | [查看](previews/coco_3.jpg) |
| coco_4.jpg | 前景可见道路 | 7.6% | [查看](previews/coco_4.jpg) |
| coco_5.jpg | 两只长颈鹿的躯干与颈部 | 7.9% | [查看](previews/coco_5.jpg) |
| coco_6.jpg | 冰箱外壳与门面 | 41.3% | [查看](previews/coco_6.jpg) |
| coco_7.jpg | 海浪与前景水面 | 45.6% | [查看](previews/coco_7.jpg) |
| coco_8.jpg | 狗的胸腹皮毛 | 9.6% | [查看](previews/coco_8.jpg) |
| coco_9.jpg | 可见沙发靠背与坐垫 | 7.0% | [查看](previews/coco_9.jpg) |
| coco_10.jpg | 摩托车黄色车身 | 7.0% | [查看](previews/coco_10.jpg) |
| eg1.png | 苹果表面 | 28.1% | [查看](previews/eg1.jpg) |
| eg2.png | 旗帜布面 | 22.5% | [查看](previews/eg2.jpg) |
| eg3.jpg | 栈桥两侧湖水 | 26.4% | [查看](previews/eg3.jpg) |
| eg4.png | 流程图内彩色图形 | 1.3% | [查看](previews/eg4.jpg) |
| eg5.png | 人物背后的山体 | 13.9% | [查看](previews/eg5.jpg) |
| eg6.png | 走势图的彩色填充区 | 6.8% | [查看](previews/eg6.jpg) |
| eg7.png | 流程图内彩色图形 | 1.3% | [查看](previews/eg7.jpg) |

文档和图表的可编辑区域较小，是为了避开正文、坐标文字及白色页面；效果尚未验证。

完整坐标、理由、文件哈希和模型尺寸见 region_specs.json 与 manifest.json。所有原图哈希检查通过。