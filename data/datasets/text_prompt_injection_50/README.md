# 图片文字提示注入样本子集

**浏览入口：[按类型看图片](浏览图片.html)**。补充查证见 [作者材料复查](SOURCE_RECHECK.md)。

整理日期：2026-11-02。这里只保存公开的攻击图片和来源记录，不运行其中的任何指令，不生成替代图凑数。

| 目录 | 类型 | 实际图片数 | 状态 |
|---|---|---:|---|
| `01_ui_task_redirection` | 界面内邮件／聊天任务诱导 | 33 | 33张独立图片，未达到50；69个任务条目存在重复图片，已去重。 |
| `02_authority_notification_spoofing` | 通知／弹窗／下载提示伪装 | 50 | 50张；按派生集 modal_popup/download_banner 标签归类，不宣称全部都冒充系统身份。 |
| `03_obfuscated_low_visibility_text` | 混淆／低可见度文字 | 50 | 50张；官方 obfuscated 标签，抽查见低对比度覆盖。不是50张已逐张验证的小字体样本。 |
| `04_visible_text_instruction` | 可见文字指令基线 | 50 | 50张；官方 embedded_text_instructions 标签。 |
| `05_distributed_cross_region` | 见最新 STATUS.md | 50 | 已收集50张。来源 RI Multimodal Injection，family=cross_region_completion 且 is_attack=true。其他研究者发布的合成图片，不是IPI作者原图。 |
| `06_qr_encoded_instruction` | 见最新 STATUS.md | 50 | 已收集50张。来源 RI Multimodal Injection，family=reachable_qr 且 is_attack=true。包括明文与二维码共存的混合载体，不承诺纯二维码，也未验证当前目标模型执行成功。 |
| `07_diagram_structured_instruction` | 见最新 STATUS.md | 0 | 已取得作者论文Figure 2，保存于 paper_examples/mindmap_paper_example.png，浏览页顶部可看。它是论文结构示例，不计入攻击数据集；仍未找到50张作者原始思维导图图片集。该文主要是安全绕过，不应直接等同间接注入。 |
| `08_chinese_multilingual` | 见最新 STATUS.md | 50 | 已收集50张土耳其语。来源 PolyOCRInject，language=tr、label=injection。目录名为保持兼容沿用，实际没有中文，不是50种语言。 |
| `09_scaling_revealed_text` | 作者缩放攻击演示图 | 3 | 原作者 Anamorpher 发布，缺47张 |
| `10_metadata_out_of_pixel_scope` | 见目录内 STATUS.md | 0 | 未收集／范围外 |

共386张数据图片，文件SHA256及解码RGB像素均为386个独立值；另有3张论文插图单列，不计入数据集。已验证全部图片可解码；不是逐张人工复核或已成功攻击的证明。分类是按来源标签映射的本地分类，类别可能有重叠，未重复拷贝到多类。这里没有良性对照集，不能单独据此估计误报率。

## 来源与使用边界

- [Meta 官方 CyberSecEval 3](https://huggingface.co/datasets/facebook/cyberseceval3-visual-prompt-injection)：后两类来源，保留官方图片原字节，MIT（数据卡声明）。`_sources/cyber_test_cases.json` 保存全部官方测试描述；manifest 仅对应选中图片。
- [TrustSight VPI-Bench 派生截图集](https://huggingface.co/datasets/tuandunghcmut/trustsight-vpibench)：前两类来源，CC-BY-4.0。它是第三方派生版本，不能写成 VPI-Bench 作者原始截图。原论文及任务集：[VPI-Bench](https://arxiv.org/abs/2506.02456)、[原始数据](https://huggingface.co/datasets/VPI-Bench/vpi-bench)。保存数据集卡于 `_sources/trust_readme.md`。
- 派生集通过 Hugging Face datasets-server 的图像预览接口下载 JPG，属于服务端图像表示，可能与原 parquet 内 PNG 编码不同。manifest 同时保存下载文件 SHA256 与来源给出的原图 SHA256，二者不要求相同。未改变图像布局、内容或叠加新的指令。
- `manifest.json` 保存每图来源、任务、目标、标签依据、文件哈希与许可；`index.csv` 便于浏览。原始截图存在上下文相关性：孤立看图可能不够，需要同时读取 manifest 中的用户任务、系统约束及攻击目标。
- `03` 的低对比度只是混淆子形式，小字体尚无专门50张子集。`05` 的跨区域形式也不能由长文本、多行排版自动替代。
- 没有下载对抗像素训练集、普通视觉误分类图片或只有攻击文本没有图片的数据。没有把 FigStep 等 jailbreak 条目混入这次基线集合。

## 复现与检索记录

采集脚本位于 `tools/dataset_collection/`。`_sources/` 保存论文网页、API 检索及数据说明，便于追溯找不到的类型。请不要把“未找到公开包”写成“该攻击不存在”。
本集合是初步研究子集，不是新的独立基准；来源与类别高度相关，后续训练/测试应按来源、原图和任务家族隔离，避免重复截图泄漏。

## 新增其他来源样本（最新）

跨区域、二维码、土耳其语、小字各50张已收集，详见 [中文读图指南](HOW_TO_READ_EXAMPLES.md)。小字在新增目录 `11_small_corner_text`。此前关于缺失的描述仅适用于原论文作者包，不能再理解成找不到任何其他公开测试图。新数据是公开合成研究测试集，非原论文作者样本；论文插图单独计数。
