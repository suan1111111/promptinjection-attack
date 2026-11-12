import json,concurrent.futures,collections
from fetch_candidates import get,root
def run(name,url):
 try:
  r=get(url);(root/(name+'.json')).write_bytes(r.content);d=r.json()
  if name.startswith('umis'):
   rows=[x['row'] for x in d['rows']];print(name,d.get('num_rows_total'),collections.Counter((r.get('family'),r.get('is_attack')) for r in rows),flush=True)
  elif name=='poly_manifest':print(name,len(d) if isinstance(d,list) else str(d)[:100])
 except Exception as e:print(name,str(e),flush=True)
jobs=[('umis_'+s,f'https://datasets-server.huggingface.co/rows?dataset=hanIshi/ri-multimodal-injection&config=image&split={s}&offset=0&length=100') for s in ['train','test','dev','external','calibration','policy']]
with concurrent.futures.ThreadPoolExecutor(6) as p:list(p.map(lambda x:run(*x),jobs))
r=get('https://huggingface.co/datasets/fevziegeyurtsevenler/PolyOCRInject/raw/main/data/manifest.jsonl');(root/'poly_manifest.jsonl').write_bytes(r.content)
rows=[json.loads(l) for l in r.text.splitlines() if l.strip()];print('poly',collections.Counter((x['language'],x['label']) for x in rows),flush=True)
