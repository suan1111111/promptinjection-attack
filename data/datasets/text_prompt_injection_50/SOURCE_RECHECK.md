# 最新补充

其他研究者公开的 RI Multimodal Injection 与 PolyOCRInject 已补入200张；另已下载IPI、Repeat-After-Me及思维导图论文插图3张，见 [中文读图指南](HOW_TO_READ_EXAMPLES.md)。以下为此前对特定原作者材料的复查历史，不代表当前仍无其他公开样本。当前数据集386张，论文插图另计。

# 回到原文和作者材料的复查

复查日期：2026-11-02。此前清单把“有文献描述一种攻击”与“存在公开图片数据集”混在一起，不够准确。下面区分实际公开内容；“未找到”仅代表本次核实结果。

| 类型 | 原始依据与公开内容 | 当前结果 |
|---|---|---|
| 跨区域文字 | [IPI III-D](https://arxiv.org/html/2603.03637v1#S3.SS4) 明确描述 Multi-Mask Distribution、按空间顺序分布指令。核对全文外链与 GitHub 按论文题目/编号搜索，未发现作者攻击图片包或生成代码链接。 | 未能收集作者原始图片，不拿普通多行覆盖图冒充跨区域样本。 |
| 二维码指令 | [MMPIBench V-D、附录A](https://arxiv.org/html/2609.09404v1) 为6种载体×4种目标，共24个攻击，其中二维码仅4个案例；附录列出载荷文字。全文没有找到公开仓库/下载链接。 | 原文确有二维码载体，但不能据此声称作者提供50张。论文该实验二维码完成攻击数为0，解码失败是重要原因。没有把文字描述渲染成作者原图。 |
| 思维导图 | [Mind Mapping Prompt Injection 的 Data Availability Statement](https://www.mdpi.com/2079-9292/14/10/1907) 指向 [ArtPrompt](https://github.com/uw-nsl/ArtPrompt) 与 [red-instruct](https://github.com/declare-lab/red-instruct)。复查仓库目录：前者是ASCII艺术攻击代码、提示数据和3张说明截图；后者是文本问答/训练材料及说明截图。 | 这些链接不是作者思维导图攻击图片包；该文主要测试有害回答绕过，不应直接当作第三方间接注入数据集。仍缺50张对应原图。 |
| 中文/多语言图片 | 此前把“现有英文规则不支持中文”列为实现缺口，没有确认对应图片数据发布。 | 不能由检测缺口推导出已有公开数据集，也不把纯文本多语言集冒充图片集。 |
| 缩放显字 | [Chameleon](https://arxiv.org/html/2512.04895v1) 本次未找到独立数据链接；它引用的 [Trail of Bits 原始研究](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems/) 明确发布 [Anamorpher](https://github.com/trailofbits/anamorpher)。 | 已下载作者仓库压缩包，保留其中3张 backend/demo_images 原图。生成器、3张正常诱饵图不是50张攻击数据；后者不纳入攻击集。 |
| 小字/分散载荷的第三方代码 | [injection-fixtures](https://github.com/munzzyy/injection-fixtures) README 声明图片通过 Pillow 本地生成，并列举 split-payload、tiny-corner 等形式；不是 IPI 作者代码。此次仓库压缩包下载返回404，未获得可固定版本的代码包。 | 是可参考的第三方构造方法，不是已下载的作者数据。没有执行或把其输出计入本集合。 |
| 大型“多模态数据集”候选 | [bordair-dataset](https://github.com/n1ps/bordair-dataset) 的 Known limitations 明确 image_content 等字段只是提取内容的文本表示，没有真实图像二进制。 | 排除，避免拿几十万条文字描述当图片数量。 |

## 本地证据

`_sources/trailofbits__anamorpher.zip`、`_sources/uw-nsl__ArtPrompt.zip` 保存核对用的作者仓库快照；它们只是来源档案，不计入图片数量。`_sources/declare-lab__red-instruct_tree.json` 保存目录清单。下载过程中 GitHub API 遇到速率限制，Anamorpher 和 ArtPrompt 改用公开 codeload 下载成功；injection-fixtures 未成功。

Anamorpher 三张原图保留于 `09_scaling_revealed_text/`，许可证、归档哈希和逐图出处已记录。全部186张是实际文件，不含缺失类别的占位图，不宣称每张均已在当前模型上成功触发注入。
