import requests,json,concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
root=Path('data/datasets/text_prompt_injection_50/_sources')
def get(url):
 s=requests.Session(); s.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1)))
 r=s.get(url,timeout=60);r.raise_for_status();return r
def job(item):
 name,url=item
 try:
  r=get(url);(root/name).write_bytes(r.content)
  if name.endswith('.json'):
   d=r.json()
   if 'siblings' in d:print(name,[(x['rfilename']) for x in d['siblings'] if x['rfilename'].endswith(('.parquet','.png','.zip'))][:18])
   elif 'rows' in d:print(name,repr(d['rows'][0])[:900])
  else:print(name,r.text[:1800])
 except Exception as e:print(name,str(e))
jobs=[('vpi_info.json','https://huggingface.co/api/datasets/VPI-Bench/vpi-bench'),('trust_readme.md','https://huggingface.co/datasets/tuandunghcmut/trustsight-vpibench/raw/main/README.md'),('vpi_rows.json','https://datasets-server.huggingface.co/rows?dataset=VPI-Bench/vpi-bench&config=default&split=test&offset=0&length=100'),('trust_rows.json','https://datasets-server.huggingface.co/rows?dataset=tuandunghcmut/trustsight-vpibench&config=default&split=upstream_attack&offset=0&length=100')]
if __name__ == '__main__':
 with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(job,jobs))
