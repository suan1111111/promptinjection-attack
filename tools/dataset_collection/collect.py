import sys,json,io,hashlib,csv,concurrent.futures
from pathlib import Path
from PIL import Image
from fetch_candidates import get
ROOT=Path('data/datasets/text_prompt_injection_50');SRC=ROOT/'_sources'
REPO='tuandunghcmut/trustsight-vpibench'
rows=[]
for offset in [0,100,200,300]:
 p=SRC/f'trust_rows_{offset}.json'
 d=get(f'https://datasets-server.huggingface.co/rows?dataset={REPO}&config=default&split=upstream_attack&offset={offset}&length=100').json()
 p.write_text(json.dumps(d,ensure_ascii=False),encoding='utf8');rows.extend(x['row'] for x in d['rows'])
print('rows',len(rows),flush=True)
groups={
 '01_ui_task_redirection': [r for r in rows if r['carrier_type'] in ['email_body','chat_message_bubble']],
 '02_authority_notification_spoofing':[r for r in rows if r['carrier_type'] in ['modal_popup','download_banner']],
}
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf8')) if (ROOT/'manifest.json').exists() else []
manifest=[r for r in manifest if r['category'] not in groups]
def fetch(item):
 group,index,r=item
 path=ROOT/group/f'{index:03d}_{r["task_id"]}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
 try:
  content=path.read_bytes() if path.exists() else get(r['image']['src']).content
  im=Image.open(io.BytesIO(content)); im.verify()
  path.write_bytes(content)
  return dict(category=group,file=str(path.relative_to(ROOT)).replace('\\','/'),task_id=r['task_id'],carrier=r['carrier_type'],platform=r['platform'],width=r['image'].get('width'),height=r['image'].get('height'),sha256=hashlib.sha256(content).hexdigest(),upstream_image_sha256=r['image_sha256'],source_dataset=REPO,source_revision='3e9e0a33150209af005ca1609c675b5640cae438',license='CC-BY-4.0 (dataset card)',user_task=r['instruction'],attack_objective=r['attack_objective'],bounding_boxes=r['verified_bounding_boxes'],label_basis='upstream carrier_type; local taxonomy mapping; no per-image ASR claim')
 except Exception as e: print('FAILED',r['task_id'],str(e),flush=True)
jobs=[]
for group,items in groups.items():
 # Alternate platforms/carriers instead of taking one contiguous block.
 items=sorted(items,key=lambda r:(r['task_id'].split('_')[-1],r['platform'],r['task_id']))
 seen=set(); selected=[]
 for r in items:
  if r['image_sha256'] in seen:continue
  seen.add(r['image_sha256']); selected.append(r)
  if len(selected)==50:break
 print(group,'available unique',len({r['image_sha256'] for r in items}),'selected',len(selected),flush=True)
 jobs.extend((group,i,r) for i,r in enumerate(selected,1))
with concurrent.futures.ThreadPoolExecutor(6) as pool:
 for result in pool.map(fetch,jobs):
  if result:manifest.append(result)
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('saved',len(manifest),'unique',len({r['sha256'] for r in manifest}),flush=True)
