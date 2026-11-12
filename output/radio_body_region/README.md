# 收音机本体区域

此次只替换可编辑区域，复用 pixel_attack_trials/optimize_qwen_pixels.py 的现有自定义掩码功能。
不改原训练内核、local_shape_fit.py 或旧实验文件。

- radio_body_mask.png：364×252 二值掩码，白色可修改、黑色不修改。
- radio_body_region_preview.png：绿色覆盖和红色边线只是区域预览，不进入模型。
- region.json：手工描绘的轮廓坐标、面积和尺寸。
- train.ps1：独立训练入口，默认沿用上一组 40 题、完整目标、四题梯度平均、1000 步、步长 1/255、epsilon 255/255、大幅随机初始化及 128 视觉 token 预算。

可编辑像素 31410，占模型输入 34.24%，约为原圆形区域的 9.9 倍。
轮廓沿收音机外壳内侧绘制，排除桌面、背景和外侧电线。未采用分割模型，边界为保守手工多边形。
使用全画幅掩码坐标 (0,0,364,252)，实际可修改范围仅由掩码白色部分确定，并非全图可修改。
不传 flower-image，因此从原图背景初始化，不粘贴参考素材颜色。

预检已通过，尚未启动训练：

```powershell
& .\output\radio_body_region\train.ps1 -ValidateOnly
```

训练入口（产生新的时间戳输出目录，不覆盖旧实验）：

```powershell
& .\output\radio_body_region\train.ps1
```

这一步不改变幅度或平滑性约束，默认仍允许大幅随机像素变化；无雪花和小幅度外观要求尚未在此步骤实现。
