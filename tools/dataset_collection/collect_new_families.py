import json,concurrent.futures,io,hashlib,collections
from pathlib import Path
from PIL import Image
from fetch_candidates import get,root as src
root=src.parent; repo='hanIshi/ri-multimodal-injection'
allrows=[]
for split in ['train','test','dev','calibration','policy']:
 first=json.loads((src/f'umis_{split}.json').read_text(encoding='utf8'))
 allrows.extend(x['row']|{'original_split':split} for x in first['rows'])
 for offset in range(100,first['num_rows_total'],100):
  p=src/f'umis_{split}_{offset}.json'
  if not p.exists():
   try:p.write_bytes(get(f'https://datasets-server.huggingface.co/rows?dataset={repo}&config=image&split={split}&offset={offset}&length=100').content)
   except Exception:
    chunks=[]
    for small_offset in [offset,offset+50]:
     try:chunks.extend(get(f'https://datasets-server.huggingface.co/rows?dataset={repo}&config=image&split={split}&offset={small_offset}&length=50').json()['rows'])
     except Exception as e:print('PAGE UNAVAILABLE',split,small_offset,str(e),flush=True)
    p.write_text(json.dumps({'rows':chunks}),encoding='utf8')
  allrows.extend(x['row']|{'original_split':split} for x in json.loads(p.read_text(encoding='utf8'))['rows'])
jobs=[]
for family,category in [('cross_region_completion','05_distributed_cross_region'),('reachable_qr','06_qr_encoded_instruction'),('small_corner_text','11_small_corner_text')]:
 rows=[r for r in allrows if r['is_attack'] and r['family']==family]
 print(family,len(rows),flush=True)
 for r in rows[:50]:jobs.append((category,r,repo))
polyrepo='fevziegeyurtsevenler/PolyOCRInject'
poly=[json.loads(l) for l in (src/'poly_manifest.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
for r in [r for r in poly if r['label']=='injection' and r['language']=='tr'][:50]:jobs.append(('08_chinese_multilingual',r,polyrepo))
def fetch(job):
 category,r,source=job
 try:
  info=json.loads((src/('umis_examples.txt' if source==repo else 'polyocr_examples.txt')).read_text(encoding='utf8'));revision=info['sha']
  media=r.get('media_path') if source==repo else r['image']
  # Prefer original PNGs where published, otherwise retain the provided viewer JPEG representation.
  published=any(x['rfilename']==media for x in info['siblings'])
  url=f'https://huggingface.co/datasets/{source}/resolve/{revision}/{media}' if published else r['image']['src']
  id=r.get('sample_id',r.get('id'));ext='.png' if published else '.jpg'
  path=root/category/(id+ext);path.parent.mkdir(exist_ok=True)
  content=path.read_bytes() if path.exists() else get(url).content
  im=Image.open(io.BytesIO(content));size=im.size;im.verify();path.write_bytes(content)
  return dict(category=category,file=str(path.relative_to(root)).replace('\\','/'),task_id=id,source_dataset=source,source_revision=revision,source_url=f'https://huggingface.co/datasets/{source}',license=info['cardData']['license'],sha256=hashlib.sha256(content).hexdigest(),width=size[0],height=size[1],image_text=r.get('media_text',r.get('text')),attack_objective=r.get('attack_goal',r.get('category')),label_basis='public synthetic research test set; upstream is_attack/family or injection/language labels; not IPI/MMPIBench author samples',original_split=r.get('original_split','test'),family=r.get('family',r.get('category')),language=r.get('language','en'),sample_kind='published_synthetic_dataset',render_recipe=r.get('render_recipe_json',r.get('render')),evidence=r.get('instruction_bearing_evidence_json'),qr_payload=r.get('qr_payload'),original_image_sha256=r.get('media_sha256',r.get('sha256')),download_representation='original' if published else 'HF viewer JPG')
 except Exception as e:print('FAILED',r.get('sample_id',r.get('id')),str(e),flush=True)
manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
categories={j[0] for j in jobs};manifest=[r for r in manifest if r['category'] not in categories]
with concurrent.futures.ThreadPoolExecutor(8) as p:
 for r in p.map(fetch,jobs):
  if r:manifest.append(r)
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('TOTAL',len(manifest),collections.Counter(r['category'] for r in manifest),flush=True)
