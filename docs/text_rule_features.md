# OCR 文字规则特征

文字规范化和五类判定函数逐字保留自：
https://github.com/Dora238/SnapGuard-anoy/blob/main/detector_image/snapguard.py
源文件 Git blob：80f282f5ea02b3ee4db177409d1ea540245d3ebd。
项目副本：backend/services/snapguard_text_matching.py；只包含依赖标准库的文字匹配函数。
继续使用项目已有 OCR，不执行作者的反色双 OCR 或 VSI 分流，因此不是整个检测器的完全复现。

已有 OCR 文本直接复用：

```powershell
.\.venv\Scripts\python.exe backend/extract_text_rules.py picture/example.png --ocr-text tmp/recognized.txt
```

仅给图片时，先检查本模块缓存；没有缓存则调用已启动的 PrivaSee OCR 后端。

```powershell
.\.venv\Scripts\python.exe backend/extract_text_rules.py picture/example.png
```

逐行转小写，仅保留 a-z、0-9 及 []()，随后执行作者的关键词、组合条件及正则匹配。
不增加中文、同义词或错字规则。纯中文被规范化删除，不受该方法覆盖。
保留原方法宽泛的 ignore/follow/alert 和任意 http/www 命中；规范化删除连字符后，
作者的 sk- 凭据规则也不能命中。未修订这些原始行为。

输出点击诱导、密码输入、链接邀请、控制指令、凭据请求五类命中行数，以及命中文字行占比。
每类每行最多命中一次，同一行可命中多类。
占比为本项目的数值化扩展：命中行原文长度之和 / 全文长度。每行只计一次，
相同内容出现在不同行时分别计入；分子不含行分隔符，分母包含全文空白及换行。
作者函数返回类别及整条命中行，不定义这个占比；该数值不是短语覆盖率。
这些数值不是攻击概率；正常操作说明、引用示例可能命中，隐晦改写和 OCR 错字可能漏检。

SQLite 默认位于 data/image_features/text_rules.sqlite3，按图片原始字节哈希保存 OCR 文本，
按文字哈希及匹配实现版本保存特征。旧规则结果保留但不复用，已有 OCR 无需重新计算。
先提交 OCR，再计算规则，后续计算失败仍保留文字。
与频率缓存文件独立，不删除或覆盖其他特征。无需额外依赖，无 JSON 输出文件，无新增界面。
OCR 失败返回非零退出码，不写入零值；成功返回空文字时输出零，仅表示没有识别出可匹配文字。
此前浏览器识别过但尚未写入此缓存的文字，需要用 --ocr-text 导入；不自动读取浏览器历史。
