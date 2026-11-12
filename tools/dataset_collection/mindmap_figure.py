import requests,concurrent.futures,json,io,hashlib
from pathlib import Path
from PIL import Image
root=Path('data/datasets/text_prompt_injection_50/paper_examples')
bases=['https://mdpi-res.com/d_attachment/electronics/electronics-14-01907/article_deploy/html/images/','https://pub.mdpi-res.com/electronics/electronics-14-01907/article_deploy/html/images/']
urls=[b+'electronics-14-01907-'+n for b in bases for n in ['g002.png','g002-550.jpg','g0A2.png','g0a2-550.jpg']]
def run(url):
 try:
  r=requests.get(url,timeout=15);r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.verify();return url,r.content
 except Exception:return None
with concurrent.futures.ThreadPoolExecutor(8) as p:results=[r for r in p.map(run,urls) if r]
if results:
 url,content=results[0];ext='.jpg' if '.jpg' in url else '.png';file='mindmap_paper_example'+ext;(root/file).write_bytes(content)
 rows=json.loads((root/'references.json').read_text(encoding='utf8'))
 rows.append(dict(file='paper_examples/'+file,caption='Mind Mapping：作者论文示例图（蓝色节点承载待完成的恶意目标）；安全绕过形式，非50张间接注入原图集',source='https://www.mdpi.com/2079-9292/14/10/1907',download=url,kind='paper_figure_not_dataset_sample',sha256=hashlib.sha256(content).hexdigest()))
 (root/'references.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8');print('saved',url)
else:print('publisher image download unavailable; retain source link')
