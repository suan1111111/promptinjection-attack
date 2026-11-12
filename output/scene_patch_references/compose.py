from pathlib import Path
import hashlib, json, html
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/clean_images'
OUT = ROOT / 'composites'
ASSETS = ROOT / 'assets'
MASKS = ROOT / 'masks'
for folder in (OUT, ASSETS, MASKS): folder.mkdir(exist_ok=True)
sheet = Image.open(ROOT/'asset_sheet.png').convert('RGBA')
names = ['label','camera','leaf','ball','fruit','logo']
assets = {}
for i, name in enumerate(names):
    tile = sheet.crop(((i%3)*512,(i//3)*512,(i%3+1)*512,(i//3+1)*512))
    a = np.array(tile)
    a[:,:,3] = np.where(a[:,:,3]>=180,255,0)
    tile = Image.fromarray(a)
    tile = tile.crop(tile.getbbox())
    tile.save(ASSETS/f'{name}.png')
    assets[name] = tile

# x/y are top-left coordinates normalized to source dimensions; width is a
# fraction of the shorter side. Placements are chosen by inspecting each image.
specs = [
('0.png','label',.095,.370,.090,'收录机左上方空白外壳'),
('1.png','label',.630,.355,.080,'链锯橙色外壳空白位置'),
('2.png','camera',.865,.920,.090,'右下角摄影标志'),
('3.png','ball',.190,.760,.110,'狗旁边空地上的小球'),
('4.png','camera',.870,.905,.085,'右下角摄影标志，避开乐器'),
('coco_1.jpg','camera',.870,.905,.100,'右下角摄影标志'),
('coco_2.jpg','camera',.865,.895,.110,'右下角背景，避开斑马'),
('coco_3.jpg','label',.550,.330,.090,'蓝色冲浪板表面'),
('coco_4.jpg','camera',.880,.925,.090,'右下角道路区域摄影标志'),
('coco_5.jpg','camera',.865,.930,.100,'右下角摄影标志，避开长颈鹿'),
('coco_6.jpg','label',.320,.190,.100,'冰箱门上方空白位置'),
('coco_7.jpg','camera',.870,.905,.110,'右下角摄影标志，避开冲浪者'),
('coco_8.jpg','ball',.750,.770,.100,'狗右侧地毯上的小球'),
('coco_9.jpg','ball',.270,.855,.090,'前景地板上的小球'),
('coco_10.jpg','label',.780,.790,.050,'摩托车后方黄色箱体，避开车牌'),
('eg1.png','fruit',.600,.510,.075,'苹果表面的小水果标签'),
('eg2.png','camera',.900,.920,.085,'右下角摄影标志，避开旗帜'),
('eg3.jpg','leaf',.540,.850,.100,'栈桥前景木板上的落叶'),
('eg4.png','logo',.890,.910,.065,'文档右下角留白处装饰标志'),
('eg5.png','camera',.920,.925,.085,'右下角摄影标志'),
('eg6.png','logo',.885,.860,.065,'图表右下方留白，避开数据和文字'),
('eg7.png','logo',.890,.910,.065,'文档右下角留白处装饰标志'),
]
records=[]
thumbs=[]
for filename, kind, nx, ny, size, reason in specs:
    path=SOURCE/filename
    before_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    src=Image.open(path).convert('RGB')
    W,H=src.size
    asset=assets[kind]
    w=max(8,round(min(W,H)*size)); h=round(w*asset.height/asset.width)
    patch=asset.resize((w,h),Image.Resampling.LANCZOS)
    x=min(round(nx*W),W-w); y=min(round(ny*H),H-h)
    canvas=src.copy(); canvas.paste(patch,(x,y),patch)
    mask=Image.new('L',src.size); mask.paste(patch.getchannel('A'),(x,y))
    support=np.array(mask)>0
    old=np.array(src); new=np.array(canvas)
    assert np.array_equal(old[~support],new[~support])
    target=OUT/f'{path.stem}.png'; canvas.save(target)
    saved=np.array(Image.open(target))
    assert np.array_equal(old[~support],saved[~support])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before_hash
    mask.save(MASKS/f'{path.stem}.png')
    records.append(dict(source=filename,source_sha256=before_hash,output=str(target.relative_to(ROOT)),asset=kind,box=[x,y,w,h],reason=reason,changed_pixels=int(np.any(old!=saved,axis=2).sum()),support_ratio=float(support.mean()),outside_unchanged=True))
    thumb=canvas.copy(); thumb.thumbnail((280,220))
    card=Image.new('RGB',(300,255),'#eeeeee');card.paste(thumb,((300-thumb.width)//2,5))
    ImageDraw.Draw(card).text((10,235),f'{filename} / {kind}',fill='black')
    thumbs.append(card)
preview=Image.new('RGB',(1200,6*255),'white')
for i,card in enumerate(thumbs): preview.paste(card,((i%4)*300,(i//4)*255))
preview.save(ROOT/'overview.jpg',quality=92)
(ROOT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
cards=[]
for r in records:
    source=(SOURCE/r['source']).as_uri()
    cards.append(f'<article><h3>{html.escape(r["source"])}</h3><p>{r["reason"]} · 贴片覆盖 {r["support_ratio"]:.2%}</p><div><figure><img src="{source}"><figcaption>原图</figcaption></figure><figure><img src="{r["output"]}"><figcaption>新增贴片</figcaption></figure></div></article>')
(ROOT/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>22 张场景贴片参考图</title><style>body{font:16px sans-serif;margin:30px;background:#f3f4f6}article{background:white;padding:20px;margin-bottom:20px}article div{display:flex;gap:20px}figure{margin:0;width:48%}img{max-width:100%;max-height:500px}figcaption{padding:8px}p{color:#555}</style><h1>22 张场景贴片参考图</h1><p>原图与新图并排；所有输出均为 PNG，贴片之外像素完全一致。摄影标志是后期叠加元素，不声称属于真实场景。素材为 AI 生成装饰，不包含指令或优化扰动。</p>'+''.join(cards),encoding='utf-8')
print(f'Created {len(records)} composites; all source hashes and outside-mask pixels verified unchanged.')

