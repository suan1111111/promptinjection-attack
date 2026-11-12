# 元指令数据公开材料

下载日期：2026-10-09

来源：https://github.com/Tingwei-Zhang/Soft-Prompts-Go-Hard
对应论文：Self-interpreting Adversarial Images（DE-FIVE 参考文献 [21]）
DE-FIVE：https://arxiv.org/html/2606.22779v1#S4.SS1.SSS1

## 文件

- Soft-Prompts-Go-Hard-main.zip：作者公开仓库的 main 分支快照。
- Soft-Prompts-Go-Hard-main/instruction_data：公开元指令问答语料与生成说明。
- Soft-Prompts-Go-Hard-main/clean_images：作者公开正常图像。
- Soft-Prompts-Go-Hard-main：保留源代码、评估脚本和作者 LICENSE；没有运行攻击生成程序或安装依赖。
- inventory.json：压缩包 SHA256、文件及问答行数统计；在下载解压校验后生成。
- vlmguard_datasets.md：VLMGuard 当前官方仓库数据说明原文。

## 完整性与范围

这是 DE-FIVE 所引用的上游元指令研究的公开材料，不能直接称为 DE-FIVE 完整实验集。
DE-FIVE 描述的测试集为 25 张正常图像和 300 张恶意元指令图像，每张搭配 20 个问题。
截至本次查找，未找到这 300 张攻击图像的独立公开下载包。VLMGuard 当前官方仓库明确不分发数据、图像或预计算特征，其新版基准与早期元指令实验也有差别。

攻击图像是否包含在上游快照中，以 inventory.json 和实际文件为准；问答 CSV 是生成或评估语料，不应误标为已经生成的恶意图像。

## 本次下载校验结果

压缩包 24,481,696 字节，CRC 校验通过，全部 30 个 PNG/JPG 图像可正常读取。
公开 instruction_data 内共有 83 个 CSV，合计 6,981 个数据行（不同文件可能重复问题，不表示独立问题数）。
clean_images 内共有 22 张正常图像和 1 个 label.txt。
其余图像为仓库其他目录中的材料，未提供 DE-FIVE 所述完整的 300 张攻击图像集合。
详细统计与压缩包 SHA256 见 inventory.json。
