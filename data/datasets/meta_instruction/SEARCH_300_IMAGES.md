# 300 张恶意元指令图像检索记录

日期：2026-10-09
结论：尚未找到 DE-FIVE / 早期 VLMGuard 使用的 300 张攻击图像原始文件公开下载地址。此前下载的代码与问答语料不满足这个目标。

已核查：
1. DE-FIVE 正文、arXiv 页面与公开代码检索。
2. VLMGuard arXiv v1 的 4.1 节与附录 C，以及现有官方代码的数据说明。
3. Soft-Prompts-Go-Hard 全部 21 次提交的完整递归文件树（无截断）。未发现 bad_prompt、output、BMP、扰动张量或攻击集压缩包。官方仓库仅有 main 分支，没有 Releases；两个 fork 的最后推送时间与上游一致。
4. Figshare 三个版本的附件元数据。最新文件与已下载 ZIP 的大小及 MD5 完全一致；旧版也均为约 24.5 MB 的同名代码包，没有独立图像附件。
5. USENIX 官方 Artifact Appendix：复现流程要求首先生成图像，再对 output/example/bad_prompt.bmp 推理。
6. 作者主页、Hugging Face 与其他公开数据托管检索：未发现目标集合。

重要解释：VLMGuard 附录 C 写明作者优化图像生成恶意数据，引用官方仓库作为优化所用数据来源。结合仓库实际内容，这不能理解为 300 张成品攻击图像已在该仓库中发布。

来源：
- https://arxiv.org/html/2606.22779v1#S4.SS1.SSS1
- https://arxiv.org/html/2410.00296v1#A3
- https://www.usenix.org/system/files/usenixsecurity25-appendix-zhang-tingwei.pdf
- https://github.com/Tingwei-Zhang/Soft-Prompts-Go-Hard
- https://api.figshare.com/v2/articles/29219051
- https://github.com/radiolab-ntu/vlmguard/blob/main/docs/DATASETS.md

可联系的作者：
- VLMGuard 第一作者 Xuefeng / Sean Du：xuefeng.du@ntu.edu.sg（https://d12306.github.io/ 当前主页）
- 上游元指令论文第一作者 Tingwei Zhang：tz362@cornell.edu（https://ztingwei.com/ 当前主页）

## 数据索取邮件草稿（未发送）

Subject: Request for the 300 adversarial meta-instruction images used in VLMGuard

Dear Professor Du,

I am studying defenses against image-based indirect prompt injection. The original VLMGuard paper (arXiv:2410.00296v1, Section 4.1 and Appendix C) describes 25 benign ImageNet images and 300 malicious meta-instruction images across Language, Politics, Formality, Spam, and Sentiment. DE-FIVE (arXiv:2606.22779) subsequently uses this benchmark.

I checked the public VLMGuard repository and the Soft-Prompts-Go-Hard GitHub/Figshare artifacts, but could only locate code, clean images, and question-answer corpora. Would you be able to share the exact 300 generated adversarial images, their corresponding clean images, question splits, meta-objective labels, and target-model/perturbation settings, or point me to their download location?

Thank you for your help.
