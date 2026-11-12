import json,csv,hashlib,collections
from pathlib import Path
from PIL import Image
root=Path('data/datasets/text_prompt_injection_50')
records=json.loads((root/'manifest.json').read_text(encoding='utf8'))
pixels=collections.defaultdict(list)
for row in records:
 p=root/row['file'];data=p.read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 with Image.open(p) as image:
  image.load();pixels[hashlib.sha256(image.convert('RGB').tobytes()+str(image.size).encode()).hexdigest()].append(row['file'])
counts=collections.Counter(r['category'] for r in records)
missing={
 '05_distributed_cross_region':'跨区域分散指令：查阅 IPI (arXiv:2603.03637) 原文与 GitHub 仓库搜索，未找到可核实的作者公开攻击图片包；不能将普通多行文字当作跨区域指令。',
 '06_qr_encoded_instruction':'二维码指令：查阅 MMPIBench (arXiv:2609.09404) 与仓库搜索，未找到公开原始图像包；论文列举载荷不等于已发布图片。不自行生成二维码凑数。',
 '07_diagram_structured_instruction':'图示／思维导图：检索 Mind Mapping Prompt Injection (Electronics 14(10):1907) 作者数据，未找到足够可核实的公开原始攻击图片；其安全绕过设定也不能直接等同间接注入。',
 '08_chinese_multilingual':'中文／多语言图片指令：本次已核实的数据以英语为主，未收集到具有明确语言标签的50张中文或多语言图片注入集；不把文本数据渲染冒充作者图片。',
 '09_scaling_revealed_text':'缩放后显现文字：本次未完成可核实原始攻击图片集收集；已有缩放检测作者仓库的示例不能扩充成50个独立攻击样本。',
 '10_metadata_out_of_pixel_scope':'EXIF 元数据指令：不属于画面文字，本次不收集。MMPIBench 涉及该载体，但未找到可直接下载的原图包。',
}
for name,reason in missing.items():
 p=root/name;p.mkdir(exist_ok=True);(p/'STATUS.md').write_text('# 未收集：0 张\n\n'+reason+'\n\n这是本次检索状态，不表示世界上不存在相关数据。\n',encoding='utf8')
rows=[('01_ui_task_redirection','界面内邮件／聊天任务诱导','33张独立图片，未达到50；69个任务条目存在重复图片，已去重。'),('02_authority_notification_spoofing','通知／弹窗／下载提示伪装','50张；按派生集 modal_popup/download_banner 标签归类，不宣称全部都冒充系统身份。'),('03_obfuscated_low_visibility_text','混淆／低可见度文字','50张；官方 obfuscated 标签，抽查见低对比度覆盖。不是50张已逐张验证的小字体样本。'),('04_visible_text_instruction','可见文字指令基线','50张；官方 embedded_text_instructions 标签。')]
text='''# 图片文字提示注入样本子集

整理日期：2026-11-02。这里只保存公开的攻击图片和来源记录，不运行其中的任何指令，不生成替代图凑数。

| 目录 | 类型 | 实际图片数 | 状态 |
|---|---|---:|---|
'''
for name,label,note in rows:text+=f'| `{name}` | {label} | {counts[name]} | {note} |\n'
for name,reason in missing.items():text+=f'| `{name}` | 见目录内 STATUS.md | 0 | 未收集／范围外 |\n'
text+=f'''
共 {len(records)} 张，文件 SHA256 独立数 {len({r['sha256'] for r in records})}，解码 RGB 像素独立数 {len(pixels)}。已验证全部图片可解码；不是逐张人工复核或已成功攻击的证明。分类是按来源标签映射的本地分类，类别可能有重叠，未重复拷贝到多类。这里没有良性对照集，不能单独据此估计误报率。

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
'''
(root/'README.md').write_text(text,encoding='utf8')
with (root/'index.csv').open('w',newline='',encoding='utf-8-sig') as f:
 fields=['category','file','task_id','source_dataset','license','sha256']; w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
(root/'validation.json').write_text(json.dumps(dict(images=len(records),files_unique=len({r['sha256'] for r in records}),decoded_pixels_unique=len(pixels),category_counts=counts,duplicates=[v for v in pixels.values() if len(v)>1]),indent=2),encoding='utf8')
print(json.dumps(dict(total=len(records),counts=counts,pixel_duplicates=[v for v in pixels.values() if len(v)>1])))
