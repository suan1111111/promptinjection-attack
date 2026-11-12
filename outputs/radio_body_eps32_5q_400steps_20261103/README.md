# 收音机本体：32/255，五题集中拟合

本组按约半小时的计算量启动，实际时长取决于 GPU 速度；400 次更新是步数限制，并非精确的定时停止。

- 背景和收音机本体掩码沿用前一组，掩码外不修改。
- epsilon 32/255，sign 步长 1/255，预算内随机初始化。
- 只训练原始前五题，保留每题完整目标答案，每步四题平均梯度。
- 400 次更新，约 1600 次训练问答处理，每题平均 320 次。
- 128 视觉 token 预算、种子 7，每 20 步保存图片及检查点。
- 训练结束后重新加载最终 PNG，用这五道训练题生成原图与扰动图回答。
- 本组不要求跨问题或跨图片泛化，不将 loss 下降当作成功。

命令：

```powershell
.\.venv\Scripts\python.exe -u pixel_attack_trials/quick_radio_body.py --epsilon 32 --steps 400 --max-train-rows 5 --run-dir outputs/radio_body_eps32_5q_400steps_20261103/run
```

训练状态看 run/training_log.csv；结果完成后看 run/result.json 和 run/training_evaluation.json。
最终图片路径为 run/adversarial_final.png。原训练内核及旧实验未改；quick_radio_body.py 仅新增可选的训练行数参数，原默认仍为 40。
