import re,json,requests,concurrent.futures
from pathlib import Path
root=Path('data/datasets/text_prompt_injection_50/_sources')
for key in ['ipi','mirage','vpi','mmpi']:
 text=(root/(key+'.txt')).read_text(encoding='utf8')
 print(key,[u for u in re.findall(r'href="([^"]+)',text) if any(s in u for s in ['github','huggingface'])])
def hf(repo):
 d=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=30).json()
 (root/(repo.replace('/','__')+'.json')).write_text(json.dumps(d),encoding='utf8')
 print(repo,d.get('description','')[:160],len(d.get('siblings',[])),[x['rfilename'] for x in d.get('siblings',[]) if x['rfilename'].endswith(('.parquet','.zip','.md'))][:12])
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(hf,['nyu-visionx/vpi','VPI-Bench/vpi-bench','tuandunghcmut/trustsight-vpibench']))
print('pyarrow',__import__('importlib.util').util.find_spec('pyarrow'))
