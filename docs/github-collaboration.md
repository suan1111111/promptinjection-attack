# GitHub 协作与完整项目还原

仓库：https://github.com/suan1111111/promptinjection-attack

源代码和文档使用 Git 管理。模型权重、数据集、图片、数据库和实验结果保存为同一仓库的 Release 附件，清单是根目录 `github-assets.json`。

## 第一次获取项目

安装 Git 和 Python 3.10 或更新版本，然后执行：

```powershell
git clone https://github.com/suan1111111/promptinjection-attack.git
cd promptinjection-attack
python tools/github_assets.py restore
```

还原命令会下载所有分卷，验证 SHA-256，然后恢复原目录。无需安装 Git LFS。请预留约 75 GiB 磁盘空间，分卷缓存和还原后的文件会同时存在。

还原程序默认不会覆盖现有文件。确认需要用发布快照替换本地附件时，才使用 `python tools/github_assets.py restore --overwrite`。下载中断后可重新执行，已经完成并通过校验的分卷不会重复下载。未完成的单个分卷会重新下载。

随后按主 README、后端和前端 README 安装依赖。虚拟环境、node_modules、下载缓存、本地工具运行时和未下载完成的模型文件不在附件中，需要通过原有安装脚本重建。不要将 `.env` 或其他本地密钥配置提交到公开仓库。

云端 API 的 benchmark 使用环境变量 `DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`；示例见 `.env.example`。前端可参考 `frontend/.env.example`。环境变量由运行脚本的终端设置；根目录 `.env` 不会被这些 benchmark 自动加载。

## 日常同步代码

保存文件只会保存到本机。完成一个可用的小功能后，通过 VS Code 的源代码管理面板提交并推送；开始新工作前拉取最新代码。

```powershell
git switch main
git pull --ff-only
git switch -c feature/my-change
# 在 VS Code 中修改并保存
git add <本次修改的文件>
git commit -m "Describe the change"
git push -u origin feature/my-change
```

在 GitHub 创建 Pull Request，检查后合并。首次上传前的旧提交历史仅在原电脑的本地分支保留，避免公开旧提交中的密钥；不要使用 `git push --all` 上传本地历史分支。

## 更新大文件

代码的 `git pull` 不会自动更新 Release 附件。附件是发布时的文件快照；模型、数据或实验结果变更后，需发布新的附件版本并更新清单。

维护者应在独立工作副本中设置 `tools/github_assets.py` 的新 `TAG`，先保留旧版 `github-assets.json` 与 `.github-transfer`，再为新版本执行 `pack`。检查新清单，将代码和清单推送到 `main` 后执行 `upload`。上传会先使用草稿 Release，全部分卷通过服务器校验后才公开；中断后执行相同命令可跳过已验证的分卷。

```powershell
python tools/github_assets.py pack
python tools/github_assets.py upload
```

打包数据库前应先停止会写入这些数据库的服务，或使用数据库提供的备份功能生成一致快照。附件还原不会自动启动数据库、安装 Python/Node 依赖或下载系统组件。
