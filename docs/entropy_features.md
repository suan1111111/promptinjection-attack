# Jedi 局部熵数值提取

来源：https://github.com/ihsenLab/jedi-CVPR2023/blob/main/Jedi-main/jedi.m
移植 get_entr_heatmap 及自编码器之前的三尺度合并、区域过滤，不运行 AE、图像修复或目标模型。
此处是 Python 移植，未用 MATLAB 执行做逐像素对照，不能声称已经验证跨语言完全一致。

项目根目录运行：

```powershell
.\.venv\Scripts\python.exe backend/extract_entropy.py "图片路径.png"
```

输出 18 个数值：每个尺度的熵均值、标准差、最大值、动态阈值、高熵面积比例（共 15 个），
以及合并过滤后的候选面积比例、区域数量和最大区域比例。比例范围为 0–1；熵单位 bit。
这些汇总是项目扩展，并非作者定义的攻击分数。熵均值统计扫描窗口而非热图像素；
标准差采用总体标准差。作者只用窗口均值调整阈值，未规定本项目标准差汇总。

遵循源码：按尺寸选三个窗口参数、半窗口步长、对称填充、包含两个端点，窗口实际边长
为参数+1；后扫描窗口覆盖重叠热图；数组按赋值范围扩展，再按 MATLAB round/索引方式裁剪。
保留预设 patch_mean/patch_std/clean_mean/clean_std 和动态阈值公式，未自行更改阈值。
8 连通区域过滤保留源码的 max(Area)>0.01 绝对像素比较，没有擅自改为面积比例。

图片解码沿用项目 EXIF 纠正及透明图白底规则，不缩放；转换为 RGB8，再按 MATLAB rgb2gray
常用系数和整数舍入得到灰度。其他位深或色彩管理输入不声称与 MATLAB 原始解码一致。
无新增训练或模型依赖，使用项目已有 NumPy/Pillow/SciPy。大图首次扫描耗时需实际测量。

共享 data/image_features/features.sqlite3，以独立配置版本保存 entropy_scale_0/1/2 和
entropy_candidates。每尺度单独提交，后续失败仍保留完成的尺度，下次只补缺失计算。
窗口熵、热图和掩膜保存在各图片配置目录的 npz/npy，不生成 JSON 文件，不修改频率、梯度
或文字缓存。原始图片字节复用同一哈希目录。

作者预设参考统计来自其实验，不保证适合当前数据。高熵不等于攻击，零候选不等于安全。
本模块提供局部贴片线索；对全图扰动的有效性需要另行验证。
