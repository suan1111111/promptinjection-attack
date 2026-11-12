import json,io,hashlib,concurrent.futures,shutil
from pathlib import Path
from PIL import Image
from fetch_candidates import get
root=Path('data/datasets/text_prompt_injection_50'); src=root/'_sources'
repo='facebook/cyberseceval3-visual-prompt-injection'
info=json.loads((src/'facebook__cyberseceval3-visual-prompt-injection.json').read_text(encoding='utf8'))
revision=info['sha']
cases=json.loads((src/'cyber_test_cases.json').read_text(encoding='utf8'))
jobs=[]
for group,tech in [('03_obfuscated_low_visibility_text',['embedded_text_instructions','obfuscated']),('04_visible_text_instruction',['embedded_text_instructions'])]:
 selected=[r for r in cases if r.get('injection_technique')==tech and r.get('image_text')][:50]
 jobs.extend((group,r) for r in selected)
def fetch(job):
 group,r=job;id=r['id'];path=root/group/f'{id:04d}.png';path.parent.mkdir(exist_ok=True)
 try:
  data=path.read_bytes() if path.exists() else get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/images/{id}.png').content
  im=Image.open(io.BytesIO(data));size=im.size;im.verify();path.write_bytes(data)
  return dict(category=group,file=str(path.relative_to(root)).replace('\\','/'),task_id=id,source_dataset=repo,source_revision=revision,source_url=f'https://huggingface.co/datasets/{repo}/blob/{revision}/images/{id}.png',license='MIT (dataset card)',sha256=hashlib.sha256(data).hexdigest(),width=size[0],height=size[1],user_task=r.get('user_input_text'),system_prompt=r.get('system_prompt'),image_text=r['image_text'],judge_question=r.get('judge_question'),injection_type=r.get('injection_type'),original_technique=r['injection_technique'],label_basis='official technique label; obfuscated is not an assertion that every image is tiny-font or low-contrast')
 except Exception as e:print('FAILED',id,str(e),flush=True)
manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
manifest=[r for r in manifest if r['category'] not in {x[0] for x in jobs}]
with concurrent.futures.ThreadPoolExecutor(6) as pool:
 for r in pool.map(fetch,jobs):
  if r:manifest.append(r)
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('total',len(manifest),'unique',len({r['sha256'] for r in manifest}),flush=True)
