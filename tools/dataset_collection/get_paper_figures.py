import json,concurrent.futures,io,hashlib
from pathlib import Path
from PIL import Image
from fetch_candidates import get
root=Path('data/datasets/text_prompt_injection_50')
out=root/'paper_examples';out.mkdir(exist_ok=True)
jobs=[('ipi_rendering.png','https://arxiv.org/html/2603.03637v1/ALL2.png','IPI Figure 4：三种低对比度文字渲染；不是跨区域数据集','https://arxiv.org/html/2603.03637v1'),('ram_teaser.png','https://arxiv.org/html/2609.04533v1/teaser.png','Repeat-After-Me：真实任务场景中的文字注入示例（论文组合图）','https://arxiv.org/html/2609.04533v1'),('mindmap_A2.png','https://mdpi-res.com/d_attachment/electronics/electronics-14-01907/article_deploy/html/images/electronics-14-01907-g0a2.png','Mind Mapping Figure A2：作者实际生成的思维导图；安全绕过形式，非独立间接注入图片','https://www.mdpi.com/2079-9292/14/10/1907')]
def run(job):
 name,url,caption,paper=job
 try:
  content=get(url).content
  im=Image.open(io.BytesIO(content));im.verify();(out/name).write_bytes(content)
  print(name,'saved',flush=True)
  return dict(file='paper_examples/'+name,caption=caption,source=paper,download=url,kind='paper_figure_not_dataset_sample',sha256=hashlib.sha256(content).hexdigest())
 except Exception as e:print(name,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(3) as p:rows=[r for r in p.map(run,jobs) if r]
(out/'references.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
