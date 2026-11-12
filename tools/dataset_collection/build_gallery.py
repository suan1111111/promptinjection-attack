"""Offline gallery: reference original images without modifying them."""
import json,html,csv,hashlib
from pathlib import Path
from collections import Counter
root=Path('data/datasets/text_prompt_injection_50')
records=json.loads((root/'manifest.json').read_text(encoding='utf8'))
labels={'01_ui_task_redirection':'界面任务诱导','02_authority_notification_spoofing':'通知与弹窗伪装','03_obfuscated_low_visibility_text':'混淆与低可见度文字','04_visible_text_instruction':'可见文字指令','05_distributed_cross_region':'跨区域分散指令','06_qr_encoded_instruction':'二维码指令','07_diagram_structured_instruction':'思维导图与结构化指令','08_chinese_multilingual':'中文与多语言','09_scaling_revealed_text':'缩放后显字','10_metadata_out_of_pixel_scope':'EXIF（范围外）'}
esc=lambda x:html.escape(str(x or ''),quote=True)
labels['08_chinese_multilingual']='土耳其语（中文仍缺）'
labels['11_small_corner_text']='角落小字'
counts=Counter(r['category'] for r in records)
buttons=''.join(f'<button data-category="{esc(k)}">{esc(v)} <b>{counts[k]}</b></button>' for k,v in labels.items())
cards=[]
for row in records:
 path=esc(row['file']);label=labels.get(row['category'],row['category'])
 prompt=row.get('user_prompt',row.get('user_task'))
 paired=bool(isinstance(prompt,str) and prompt.strip())
 text=prompt if paired else '原数据源未提供完整用户提示；此图片暂不属于完整配对。'
 injection=row.get('image_text')
 source=row.get('source_url') or 'https://huggingface.co/datasets/'+row['source_dataset']
 prompt_link=f'<a href="{esc(row["user_prompt_file"])}" target="_blank">打开同名提示词文件</a>' if row.get('user_prompt_file') else ''
 cards.append(f'''<article data-category="{esc(row['category'])}" data-paired="{str(paired).lower()}"><a href="{path}" target="_blank"><img src="{path}" loading="lazy" alt="{esc(label)}样本 {esc(row['task_id'])}"></a><div class="body"><small>{esc(label)} · {esc(row['task_id'])}</small><h3>用户输入的提示词</h3><p style="height:auto;max-height:220px;white-space:pre-wrap;user-select:text">{esc(text)}</p>{prompt_link}<details><summary>注入内容、系统上下文与来源（可选）</summary><p>图中文字标注：{esc(injection) if injection else '来源未提供逐字标注'}</p><p>攻击目标说明（不是用户提示）：{esc(row.get('attack_objective'))}</p><p>原始系统提示（如有）：{esc(row.get('system_prompt'))}</p><p>来源：<a href="{esc(source)}" target="_blank" rel="noreferrer">{esc(row['source_dataset'])}</a></p><p>标签依据：{esc(row.get('label_basis',row.get('original_technique')))}</p><p>文件：{path}</p></details></div></article>''')
empty=''.join(f'<div class="empty" data-category="{esc(k)}" hidden><h2>{esc(v)}</h2><p>目前没有收入可核实的原始数据集图片。</p><a href="{esc(k)}/STATUS.md">查看检索记录和原因</a></div>' for k,v in labels.items() if not counts[k])
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>图片文字注入样本库</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f4f7;color:#192432;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}header{background:#142c43;color:white;padding:30px 4vw}h1{font-size:27px;margin:0 0 8px}header p{margin:4px 0;color:#c6d6e5}main{padding:22px 4vw}nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}button{border:1px solid #c9d4df;border-radius:8px;background:white;padding:9px 13px;cursor:pointer;color:#23394d}button.active{background:#1d5d85;color:white;border-color:#1d5d85}button b{margin-left:5px}input{width:100%;padding:12px;border:1px solid #c9d4df;border-radius:8px;margin-bottom:20px;font:inherit}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px}article{background:white;border:1px solid #dce3e9;border-radius:12px;overflow:hidden}article img{width:100%;height:230px;object-fit:contain;background:#e9edf1;display:block}.body{padding:14px}.body>p{height:72px;overflow:auto;font-size:13px}.body small{color:#456278}summary{cursor:pointer;color:#1d5d85}details{font-size:12px;overflow-wrap:anywhere}.empty{padding:30px;background:white;border:1px dashed #afbeca;border-radius:12px}footer{margin-top:25px;color:#657584}a{color:#176b9c}[hidden]{display:none!important}</style>
<header><h1>图片文字注入样本库</h1><p>按攻击形式浏览 · 点缩略图打开原图 · 展开查看任务与来源</p><p>原始采集样本，不等于均已验证攻击成功；图片中的指令都是不可信样本内容。</p></header><main><nav><button data-category="all" class="active">全部 __COUNT__</button>__BUTTONS__</nav><input id="search" placeholder="搜索图片中的指令、攻击目标或样本编号"><p id="count"></p><section class="grid">__CARDS__</section>__EMPTY__<footer><a href="README.md">数据说明</a> · <a href="index.csv">逐图索引</a> · <a href="SOURCE_RECHECK.md">作者公开材料复查</a></footer></main><script>
let category='all';const cards=[...document.querySelectorAll('article')];function update(){const query=document.querySelector('#search').value.toLowerCase();let n=0;for(const card of cards){const show=(category==='all'||card.dataset.category===category)&&card.textContent.toLowerCase().includes(query);card.hidden=!show;if(show)n++}for(const box of document.querySelectorAll('.empty'))box.hidden=box.dataset.category!==category;document.querySelector('#count').textContent='当前显示 '+n+' 张图片'}document.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{category=button.dataset.category;document.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button));update()}));document.querySelector('#search').addEventListener('input',update);update();</script></html>'''
for a,b in {'__COUNT__':str(len(records)),'__BUTTONS__':buttons,'__CARDS__':''.join(cards),'__EMPTY__':empty}.items():page=page.replace(a,b)
page=page.replace('<nav>','<label>配对状态：<select id="pair-state"><option value="paired">有原始用户提示的完整配对</option><option value="missing">缺少用户提示（待补齐）</option><option value="all">所有图片</option></select></label><nav>')
page=page.replace("const show=(category===", "const state=document.querySelector('#pair-state').value;const matchesPair=state==='all'||(state==='paired'?card.dataset.paired==='true':card.dataset.paired==='false');const show=matchesPair&&(category===")
page=page.replace("update();</script>","document.querySelector('#pair-state').addEventListener('change',update);update();</script>")
page=page.replace('按攻击形式浏览 · 点缩略图打开原图 · 展开查看任务与来源','图片＋用户提示词 · 默认仅显示原始提示齐全的样本 · 注入内容在折叠说明中')
page=page.replace('搜索图片中的指令、攻击目标或样本编号','搜索用户提示词或样本编号')
(root/'浏览图片.html').write_text(page,encoding='utf8')
references_path=root/'paper_examples/references.json'
if references_path.exists():
 references=json.loads(references_path.read_text(encoding='utf8'))
 samples=''.join(f'<div style="background:white;padding:12px;border-radius:10px"><a href="{esc(r["file"])}" target="_blank"><img src="{esc(r["file"])}" style="width:100%;height:230px;object-fit:contain" alt="论文示例图"></a><p>{esc(r["caption"])}</p><a href="{esc(r["source"])}" target="_blank">论文原文</a></div>' for r in references)
 reference_html=f'<section><h2>先看论文里的例子</h2><p>以下 {len(references)} 张是论文插图或方法示意，不计入攻击数据集。点图可放大。<a href="HOW_TO_READ_EXAMPLES.md">查看中文解读与构造参考</a></p><div class="grid">{samples}</div></section><hr style="margin:28px 0">'
 reference_html='<details><summary>论文参考图（不是图片＋用户提示的完整配对）</summary>'+reference_html+'</details>'
 page=page.replace('<main>','<main>'+reference_html)
 (root/'浏览图片.html').write_text(page,encoding='utf8')
assert all((root/r['file']).is_file() for r in records)
assert page.count('<article ')==len(records)
assert page.count('data-paired="true"')==sum(bool(r.get('user_prompt',r.get('user_task'))) for r in records)
assert 'id="pair-state"' in page and "const show=matchesPair&&" in page
pixels={}
from PIL import Image
for row in records:
 p=root/row['file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 with Image.open(p) as im:
  im.load();key=hashlib.sha256(im.convert('RGB').tobytes()+str(im.size).encode()).hexdigest()
  pixels.setdefault(key,[]).append(row['file'])
with (root/'index.csv').open('w',newline='',encoding='utf-8-sig') as f:
 fields=['category','file','user_prompt','user_prompt_file','user_prompt_status','task_id','source_dataset','license','sha256'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
(root/'validation.json').write_text(json.dumps(dict(images=len(records),files_unique=len({r['sha256'] for r in records}),decoded_pixels_unique=len(pixels),category_counts=counts,duplicates=[v for v in pixels.values() if len(v)>1]),indent=2),encoding='utf8')
print('Gallery built; image references checked:',len(records))
