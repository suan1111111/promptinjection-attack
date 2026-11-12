import json
from pathlib import Path
root=Path('data/datasets/text_prompt_injection_50')
notes={
'05_distributed_cross_region':'已收集50张。来源 RI Multimodal Injection，family=cross_region_completion 且 is_attack=true。其他研究者发布的合成图片，不是IPI作者原图。',
'06_qr_encoded_instruction':'已收集50张。来源 RI Multimodal Injection，family=reachable_qr 且 is_attack=true。包括明文与二维码共存的混合载体，不承诺纯二维码，也未验证当前目标模型执行成功。',
'08_chinese_multilingual':'已收集50张土耳其语。来源 PolyOCRInject，language=tr、label=injection。目录名为保持兼容沿用，实际没有中文，不是50种语言。',
'11_small_corner_text':'已收集50张。来源 RI Multimodal Injection，family=small_corner_text 且 is_attack=true。属于其他研究者公开的合成测试集。',
'07_diagram_structured_instruction':'已取得作者论文Figure 2，保存于 paper_examples/mindmap_paper_example.png，浏览页顶部可看。它是论文结构示例，不计入攻击数据集；仍未找到50张作者原始思维导图图片集。该文主要是安全绕过，不应直接等同间接注入。',
}
for name,text in notes.items():
 p=root/name;p.mkdir(exist_ok=True);(p/'STATUS.md').write_text('# 最新状态\n\n'+text+'\n\n详细出处和外观说明见 ../HOW_TO_READ_EXAMPLES.md。\n',encoding='utf8')
readme=root/'README.md'
text=readme.read_text(encoding='utf8')
text=text.replace('共 186 张（原采集183张＋作者演示3张），文件 SHA256 独立数186。','共386张数据图片，文件SHA256及解码RGB像素均为386个独立值；另有3张论文插图单列，不计入数据集。')
for name,note in notes.items():
 lines=text.splitlines()
 for i,l in enumerate(lines):
  if l.startswith('| `'+name+'`'):lines[i]=f'| `{name}` | 见最新 STATUS.md | {0 if name.startswith("07") else 50} | {note} |'
 text='\n'.join(lines)+'\n'
text+='\n## 新增其他来源样本（最新）\n\n跨区域、二维码、土耳其语、小字各50张已收集，详见 [中文读图指南](HOW_TO_READ_EXAMPLES.md)。小字在新增目录 `11_small_corner_text`。此前关于缺失的描述仅适用于原论文作者包，不能再理解成找不到任何其他公开测试图。新数据是公开合成研究测试集，非原论文作者样本；论文插图单独计数。\n'
readme.write_text(text,encoding='utf8')
p=root/'SOURCE_RECHECK.md';text=p.read_text(encoding='utf8');text='# 最新补充\n\n其他研究者公开的 RI Multimodal Injection 与 PolyOCRInject 已补入200张；另已下载IPI、Repeat-After-Me及思维导图论文插图3张，见 [中文读图指南](HOW_TO_READ_EXAMPLES.md)。以下为此前对特定原作者材料的复查历史，不代表当前仍无其他公开样本。当前数据集386张，论文插图另计。\n\n'+text;p.write_text(text,encoding='utf8')
for r in json.loads((root/'paper_examples/references.json').read_text(encoding='utf8')):
 assert (root/r['file']).is_file()
print('README, status and source notes updated')
