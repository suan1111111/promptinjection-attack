import concurrent.futures,json,re
from fetch_candidates import get,root
jobs={
 'polyocr':'https://huggingface.co/api/datasets/fevziegeyurtsevenler/PolyOCRInject',
 'umis':'https://huggingface.co/api/datasets/hanIshi/ri-multimodal-injection',
 'fcm':'https://huggingface.co/api/datasets/QFIN/FCMBench-Data',
 'repeat':'https://arxiv.org/html/2609.04533v1',
 'mindmap':'https://www.mdpi.com/2079-9292/14/10/1907',
 'visinject':'https://huggingface.co/datasets/jeffliulab/visinject/raw/main/README.md',
}
def run(item):
 k,url=item
 try:
  r=get(url);(root/(k+'_examples.txt')).write_bytes(r.content)
  if k in ['polyocr','umis','fcm']:
   d=r.json();print(k,str(d.get('cardData'))[:2400].encode('ascii','backslashreplace').decode(),[x['rfilename'] for x in d.get('siblings',[])][:12],flush=True)
  else:print(k,'links',[u for u in re.findall(r'href="([^"]+)',r.text) if any(t in u for t in ['github','huggingface'])],'images',re.findall(r'<img[^>]+src="([^"]+)',r.text)[:15],flush=True)
 except Exception as e:print(k,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(6) as p:list(p.map(run,jobs.items()))
for k in ['ipi','mmpi']:
 s=(root/(k+'.txt')).read_text(encoding='utf8')
 print(k,'images',re.findall(r'<img[^>]+src="([^"]+)',s),flush=True)
