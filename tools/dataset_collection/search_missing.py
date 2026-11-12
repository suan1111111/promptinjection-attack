import json,concurrent.futures
from fetch_candidates import get,root
queries=[('hf','visual prompt injection'),('hf','prompt injection'),('gh','"2605.28116"'),('gh','"2603.03637"'),('gh','"Mind Mapping Prompt Injection"'),('gh','"steganographic prompt"'),('gh','"MMPIBench"')]
def run(item):
 kind,q=item
 import urllib.parse
 url=('https://huggingface.co/api/datasets?search='+urllib.parse.quote(q)+'&limit=100') if kind=='hf' else ('https://api.github.com/search/repositories?q='+urllib.parse.quote(q))
 try:
  r=get(url);d=r.json();(root/(kind+'_search_'+str(queries.index(item))+'.json')).write_text(json.dumps(d),encoding='utf8')
  print(q,[(x.get('id') or x.get('full_name'),x.get('description','')[:90]) for x in (d if kind=='hf' else d.get('items',[]))][:30],flush=True)
 except Exception as e: print(q,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(run,queries))
