# 图片与用户提示词配对

2026-11-03 起以“图片＋原数据源提供的用户输入提示词”为完整样本标准。攻击内容、攻击目标、预期答案、任务ID均不能代替用户提示词。

- 完整配对：183组，其中83组来自 TrustSight VPI-Bench（instruction字段），100组来自Meta CyberSecEval 3（user_input_text字段）。原始提示逐字保存，没有把网址型GUI任务改写成普通图像问答。
- 待补齐：203张，其中RI 150张、PolyOCRInject 50张、Anamorpher演示3张。RI数据说明明确说只提供trusted_task_id、task_topic、gold_task_answer，不含完整trusted-task prompt。PolyOCRInject公开标注是画面文字，不能据此伪造用户提问。Anamorpher演示没有逐图原始用户提问。
- 浏览页默认只显示完整配对，图片下方为用户提示；注入内容及系统提示在折叠说明里。可以切换“缺少用户提示”检查待补齐图片。论文图独立折叠，不充作完整样本。
- 每组完整配对都有图片旁的同名`.prompt.txt`文件，UTF-8，无附加标题，内容就是原始用户提示。`paired_samples.jsonl`和`index.csv`记录路径与来源；缺失项在`missing_user_prompts.csv`。
- 没有删除待补齐图片，没有将本项目自编的评估提示冒充原始用户输入。仅凭图片不能恢复原实验用户提示。来源提供系统上下文时仍保留，评估攻击是否成功可能需要该上下文。

来源核对文件：`_sources/umis_dataset_readme.md`、原始`cyber_test_cases.json`与`trust_rows_*.json`。
