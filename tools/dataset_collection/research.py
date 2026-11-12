import requests, concurrent.futures, json
from pathlib import Path
OUT=Path('data/datasets/text_prompt_injection_50/_sources'); OUT.mkdir(parents=True,exist_ok=True)
urls={
 'ipi':'https://arxiv.org/html/2603.03637v1',
 'mirage':'https://arxiv.org/html/2605.28116v1',
 'vpi':'https://arxiv.org/html/2506.02456',
 'mmpi':'https://arxiv.org/html/2609.09404v1',
 'hf_vpi':'https://huggingface.co/api/datasets?search=VPI&limit=100',
 'hf_mirage':'https://huggingface.co/api/datasets?search=MIRAGE&limit=100',
 'gh_ipi':'https://api.github.com/search/repositories?q=%22Image-based+Prompt+Injection%22',
 'gh_mmpi':'https://api.github.com/search/repositories?q=MMPIBench',
 'gh_vpi':'https://api.github.com/search/repositories?q=VPI-Bench',
}
def fetch(item):
 k,u=item
 try:
  r=requests.get(u,timeout=45); (OUT/(k+'.txt')).write_text(r.text,encoding='utf8')
  if k.startswith('gh'): print(k,r.status_code,[(x['full_name'],x.get('description')) for x in r.json().get('items',[])][:8])
  elif k.startswith('hf'): print(k,[x['id'] for x in r.json()])
  else:
   import re
   print(k,r.status_code,sorted(set(re.findall(r'https?://[^\s<>"\\]+(?:github|huggingface)[^\s<>"\\]*',r.text)))[:20])
 except Exception as e: print(k,str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool: list(pool.map(fetch,urls.items()))
