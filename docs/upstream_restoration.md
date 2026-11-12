# PrivaSee 原项目功能恢复记录

对照仓库：https://github.com/729149195/PrivaSee （main，2026-10-07 下载源码）。
本地原始 HEAD 8b0e5b6 与下载源码的既有业务文件一致，未发现缺少上游源码文件。

## 恢复与保留

- 恢复模型设置导入、图片提取、历史设置和模型显示工具的上游实现。
- 原默认模型仍为上游 Qwen3/Gemma3；本机选择通过 `frontend/.env.local` 覆盖。
  `.env.example` 提供可提交的配置样例。图片模型为 Qwen2.5-VL-3B；Qwen2.5-3B 保留。
  已安装 Qwen3-4B-Instruct-2507 作为本机默认隐私流程模型，使用 NF4 适配本机 8 GB 显卡。
- 补充本地 `/v1/chat/completions` 的视觉模型分支，保留真正的图片输入，不再静默丢弃图片。
  文本模型收到图片请求时返回明确错误。未知模型名返回错误，不冒充其他模型。
- 恢复 OCR 的逐段输出。上游使用 `infer(custom_streamer=...)`，本机官方模型不接受此参数；
  独立兼容模块将 streamer 传入 generate，推理结束后恢复原方法，不修改下载模型源码。
- Whisper 使用项目现有 `base.pt` 与原生 openai-whisper，恢复语言自动识别和原分段接口。
  项目内准备名为 ffmpeg 的可执行文件，以支持原生 Whisper 的音频解码。
- DeepSeek 官方模型依赖 Transformers 4.46.3；Qwen2.5-VL 使用 4.57.6。
  `.venv-ocr` 仅安装旧版 transformers/tokenizers，并通过 pth 共用主环境其他依赖及模型权重。
  pth 使用 site.addsitedir，确保共享 pywin32 的模块和 DLL 启动配置也生效。
  `start-all.ps1` 使用主环境启动 Qwen 服务，使用 OCR 环境启动原后端，避免修改模型源码。
- 保留 Windows Office 转 PDF、项目内 Poppler、模型路径、CPU 嵌入和 Whisper 按需加载等必要适配。
- OCR 的 PyTorch 和 streamer 改为识别时按需加载，普通接口及健康检查无需初始化推理环境。
  健康检查的 device 为 not_initialized 时表示尚未加载运行时，不代表服务不可用。
  PowerPoint 转换结束使用无参数 Close，保留用户已有演示窗口，并确保失败时仍释放 COM 资源。
  Office 自动化的临时忙状态进行有限重试，其他错误直接返回。
- 保留使用环境变量配置云 API 密钥的做法，未恢复仓库内硬编码凭据。
- 频域提取、SQLite 增量缓存、终端输出、测试均保留；服务包继续按需导入，避免频域 CLI 加载大模型。

## 并非本地删掉的功能

上游账号仍为浏览器本地登录；`useImageAnalysis` 仍是上传状态管理。
实际图片信息提取在 infonSlice，图片内容分析在 configSlice。此次不另造后端账号系统，
也不把尚未规划的结构化 OCR 特征当作原系统缺失项。

## 使用

运行 `./start-all.ps1` 启动本机文本/视觉兼容服务、原 Flask 后端和原前端。
重建环境时，先准备 `.venv` 并安装 `backend/requirements.txt`，再运行 `./setup-ocr-runtime.ps1`。
已有浏览器用户保存的模型设置优先于默认值。若此前选中了 deepseek-ocr 作为图片解析模型，
请在模型设置中将图片解析选择为 `qwen2.5-vl-3b-instruct`；OCR 文档命令仍选择 deepseek-ocr。
信息提取、隐私推理和保护建议选择 `qwen3-4b-instruct-2507`；轻量对话仍可选 `qwen2.5-3b-instruct`。
已有用户需要手动更新保存的模型选择；环境默认值不会覆盖浏览器保存的设置。
其他上游 Ollama 模型需要运行真实 Ollama 并安装相应模型；当前本地兼容服务提供上述三种模型。
云模型需要在设置中填入自己的 API 密钥。

三种本地模型按请求切换，首次使用要加载权重；GPU 上视觉模型及 Qwen3 使用 NF4。
这三个模型在本地 11434 端口的提取请求允许等待 20 分钟，并避免被页面一分钟清理提前中断。
可以通过 VITE_LOCAL_MODEL_TIMEOUT_MS 调整等待上限，其他服务保持原超时策略。
Qwen3 权重可通过 `.\.venv\Scripts\python.exe backend/download_privacy_model.py` 重新下载。
视觉输入使用浏览器上传的 base64 data URL，不接受服务器任意文件路径或远程图片 URL。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
.\.venv-ocr\Scripts\python.exe -m unittest discover -s backend/tests -v
cd frontend
& '..\tools\node\node.exe' '.\node_modules\vite\bin\vite.js' build
```

自动验证不代表所有模型输出的准确性均已确认。OCR/语音/Office、完整隐私流程的真实输入验证
与配置可用性须分别记录，不能只根据健康检查判定已跑通。

当前验证记录：前端生产构建通过，20 项自动测试通过（含原频域数值、缓存恢复、
OCR streamer 两种接口、图像输入保留及本地模型接口）。真实 Qwen 视觉模型正确读取
测试图片中的姓名和年龄；文本模型流式输出姓名、年龄；记忆流真实写入及相似检索通过。
频域 CLI 的真实图片缓存复用、FFmpeg 解码程序可执行性、项目 Poppler 对真实 PDF 首页面转换通过。

OCR 真实加载发现 matplotlib 缺失及新 Transformers 删除旧 LlamaAttention 接口，
已分别补齐依赖及配置官方 4.46.3 独立环境。新增完整后端按需启动检查，两套环境各 20 项测试通过。
本次真实模型验证中 PyTorch 首次初始化耗时 865.9 秒，随后正常识别 RTX 3060 Ti。
Whisper 真实上传接口通过：自动检测英语，返回 “My name is Alice. I am 28 years old.”，
且返回真实分段时间。首次 DLL 初始化可能耗时较长；现在不会阻塞后端普通接口启动。
OCR 真实上传 SSE 接口通过：逐段输出 Name: / Alice. / Age: / 28.，最终正确识别测试图片。
Word、Excel、PowerPoint 的真实文档转 PDF 均通过，三种格式连续转换也通过。

完整隐私流程测试发现 Qwen2.5-3B 在中英文示例中漏报姓名，且会错误拆分某些信息元。
因此它可以保留作轻量对话模型，但不能仅凭接口通畅认定隐私效果满足要求。
Qwen3 中英文真实样本已完成信息元提取、风险推理及三档保护建议的数据流验证。
这不代表隐私效果已达标：实际仍出现漏报姓名、风险类别误判，以及低保护文本保留地址却声称已移除。
提示词强化了五字段格式；完整建议行若字段数不符，会被拒绝解析，避免静默截断正文。
因此格式不合规时可能少于三条建议，仍需更多样本评估或使用更强模型。云模型须使用用户自己的密钥。
信息元地址中的未转义逗号现在按末尾类型及置信度边界恢复，保留完整地址数值。
前端 9 项解析及等待策略测试通过，生产构建通过。频域代码与攻击图片训练代码未修改。
