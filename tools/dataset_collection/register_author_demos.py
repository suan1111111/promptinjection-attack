import json,hashlib,zipfile
from pathlib import Path
from PIL import Image
root=Path('data/datasets/text_prompt_injection_50')
z=zipfile.ZipFile(root/'_sources/trailofbits__anamorpher.zip')
archive_sha=hashlib.sha256((root/'_sources/trailofbits__anamorpher.zip').read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
manifest=[r for r in manifest if r['category']!='09_scaling_revealed_text']
for method in ['nearest','bilinear','bicubic']:
 name=method+'_demo.png';file='09_scaling_revealed_text/'+name;p=root/file
 with Image.open(p) as im:size=im.size;im.verify()
 manifest.append(dict(category='09_scaling_revealed_text',file=file,task_id=name,source_dataset='trailofbits/anamorpher',source_url='https://github.com/trailofbits/anamorpher/blob/main/backend/demo_images/'+name,source_revision='main archive SHA256 '+archive_sha,license='Apache-2.0; see saved license',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),width=size[0],height=size[1],attack_objective='作者仓库原始缩放攻击演示图；不是50张独立攻击数据集。',label_basis='author backend/demo_images; original bytes; not individually ASR-tested locally',sample_kind='author_demo',scaling_algorithm=method))
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(root/'_sources/Anamorpher_LICENSE.txt').write_bytes(z.read('anamorpher-main/LICENSE'))
status='''# 已补到作者演示原图：3 张（缺 47 张）

来源：Trail of Bits 原作者仓库 https://github.com/trailofbits/anamorpher ，backend/demo_images 下三个 PNG，原字节保存。Apache-2.0 许可证另存于 _sources/Anamorpher_LICENSE.txt。

nearest_demo.png、bilinear_demo.png、bicubic_demo.png 是作者用于演示相应缩放方法的图片，尚未在本项目逐个验证其隐藏载荷内容和攻击成功率。必须匹配缩放库、算法、目标尺寸，不能仅凭浏览器显示效果判断攻击是否出现。

仓库还发布生成代码和正常诱饵图；正常诱饵图未充作攻击样本。没有发现作者现成的50张独立攻击图像集，未运行生成器凑数。论文 Chameleon (2512.04895) 本次也未找到其独立数据发布地址。
'''
(root/'09_scaling_revealed_text/STATUS.md').write_text(status,encoding='utf8')
readme=(root/'README.md').read_text(encoding='utf8').replace('| `09_scaling_revealed_text` | 见目录内 STATUS.md | 0 | 未收集／范围外 |','| `09_scaling_revealed_text` | 作者缩放攻击演示图 | 3 | 原作者 Anamorpher 发布，缺47张 |').replace('共 183 张，文件 SHA256 独立数 183，解码 RGB 像素独立数 183。','共 186 张（原采集183张＋作者演示3张），文件 SHA256 独立数186。').replace('# 图片文字提示注入样本子集','# 图片文字提示注入样本子集\n\n**浏览入口：[按类型看图片](浏览图片.html)**。补充查证见 [作者材料复查](SOURCE_RECHECK.md)。')
(root/'README.md').write_text(readme,encoding='utf8')
print('Registered 3 author demos; total',len(manifest))
