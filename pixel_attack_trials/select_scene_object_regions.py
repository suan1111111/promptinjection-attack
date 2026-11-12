"""Prepare reviewed scene-region masks and previews; never trains any model.

Polygons are hand-selected normalized source-image coordinates. They are
conservative region proposals, not pixel-perfect semantic segmentation.
"""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/clean_images'
OUT = ROOT / 'output/scene_object_regions_v1'

# Each entry selects existing scene content. No external sticker is inserted.
SPECS = [
dict(source='0.png', title='收音机本体', mode='approved_radio', reason='沿用已经确认的收音机区域，背景和外侧电线不编辑。'),
dict(source='1.png', title='链锯外壳与导板', reason='选择链锯实体，避开上方手柄空洞及工作台。', polygons=[
[(.58,.36),(.64,.34),(.71,.33),(.80,.34),(.83,.43),(.86,.55),(.79,.65),(.69,.65),(.61,.60),(.57,.50)],
[(.063,.478),(.09,.465),(.50,.445),(.53,.48),(.52,.55),(.11,.57),(.065,.55)]]),
dict(source='2.png', title='教堂两侧塔楼墙面', reason='选择建筑墙面，避开天空和前景雕像。', polygons=[
[(.075,.33),(.28,.275),(.38,.39),(.30,.54),(.12,.69),(.02,.90),(0,.90),(0,.55)],
[(.64,.20),(.90,.14),(.97,.61),(.95,.82),(.70,.75),(.66,.47)]]),
dict(source='3.png', title='狗的躯干与腿部', mode='light_fur', reason='选择白色躯干和腿部，保留头脸，避开地板。', polygons=[
[(.452,.40),(.565,.405),(.63,.48),(.68,.58),(.71,.70),(.69,.78),(.61,.82),(.58,.88),(.53,.88),(.53,.81),(.49,.81),(.43,.88),(.37,.88),(.38,.83),(.41,.77),(.43,.63)]]),
dict(source='4.png', title='圆号金属表面', mode='gold', reason='在圆号轮廓内选择金色金属，尽量排除管道间白色空隙。', polygons=[
[(.60,.04),(.65,.19),(.78,.45),(.70,.46),(.72,.62),(.69,.79),(.57,.91),(.38,.94),(.25,.82),(.22,.62),(.30,.43),(.43,.32),(.55,.21)]]),
dict(source='coco_1.jpg', title='击球者球衣', reason='选择球衣主体，尽量避开脸部、球棒和背景草地。', polygons=[
[(.91,.31),(.96,.305),(.99,.37),(.99,.78),(.98,.82),(.93,.82),(.83,.85),(.815,.70),(.805,.58),(.78,.52),(.80,.49),(.83,.48),(.845,.43),(.87,.38)]],
exclude_ellipses=[(.79,.26,.90,.40)]),
dict(source='coco_2.jpg', title='前景斑马躯干条纹区域', reason='选择前景斑马的躯干和颈后区域，保留眼睛、口鼻和背景。', polygons=[
[(0,.45),(.08,.46),(.12,.41),(.20,.37),(.32,.32),(.41,.26),(.46,.24),(.49,.31),(.48,.40),(.46,.55),(.43,.64),(.39,.74),(.34,.85),(.29,1),(0,1)]]),
dict(source='coco_3.jpg', title='中间两块蓝色冲浪板', reason='沿板面轮廓选择现有冲浪板，排除沙地。', polygons=[
[(.49,.005),(.612,.005),(.63,.12),(.651,.30),(.69,.58),(.715,.78),(.713,.82),(.70,.835),(.68,.827),(.66,.79),(.63,.68),(.60,.55),(.57,.40),(.535,.21)],
[(.65,.005),(.687,.005),(.72,.16),(.75,.38),(.79,.60),(.808,.74),(.80,.78),(.78,.785),(.763,.765),(.74,.68),(.70,.50),(.67,.27)]]),
dict(source='coco_4.jpg', title='前景可见道路', reason='选择车辆两侧空出的路面，避开主要人物和车辆。', polygons=[
[(.19,.82),(.32,.80),(.48,.80),(.49,.91),(.52,.99),(.17,.99),(.15,.91)],
[(.78,.85),(.87,.83),(.88,.98),(.76,.98),(.76,.91)]]),
dict(source='coco_5.jpg', title='两只长颈鹿的躯干与颈部', reason='选择动物皮毛实体，避开头部和腿间背景。', polygons=[
[(.15,.34),(.20,.33),(.25,.36),(.34,.40),(.47,.44),(.55,.48),(.55,.57),(.49,.60),(.38,.55),(.30,.47),(.23,.42),(.16,.37)],
[(.33,.30),(.38,.28),(.43,.32),(.53,.35),(.65,.40),(.74,.44),(.76,.54),(.72,.56),(.61,.54),(.53,.46),(.47,.39),(.38,.34)]],
exclude_ellipses=[(.13,.31,.22,.37),(.32,.26,.41,.32)]),
dict(source='coco_6.jpg', title='冰箱外壳与门面', reason='选择冰箱本体，排除门框、房间和地板。', polygons=[
[(.06,.18),(.25,.13),(.61,.16),(.64,.84),(.34,.94),(.27,.90),(.07,.83)]]),
dict(source='coco_7.jpg', title='海浪与前景水面', reason='沿浪区选择水面，排除天空并留出冲浪者。', polygons=[
[(0,.50),(.10,.44),(.19,.44),(.26,.48),(.40,.54),(.60,.56),(.80,.60),(1,.61),(1,1),(0,1)]], exclude=[
[(.22,.55),(.25,.54),(.28,.57),(.28,.64),(.30,.65),(.23,.66),(.21,.62)]]),
dict(source='coco_8.jpg', title='狗的胸腹皮毛', reason='选择胸腹部的连续皮毛区域，保留整个头脸，排除地毯。', polygons=[
[(.34,.60),(.44,.61),(.53,.61),(.61,.57),(.64,.60),(.62,.77),(.58,.90),(.52,.96),(.43,.95),(.36,.87),(.30,.74),(.29,.65)]]),
dict(source='coco_9.jpg', title='可见沙发靠背与坐垫', reason='选择沙发面料，尽量避开人物和脸部。', polygons=[
[(.42,.36),(.52,.34),(.55,.36),(.53,.49),(.56,.55),(.53,.60),(.43,.59),(.39,.56),(.41,.48)],
[(.32,.60),(.42,.61),(.54,.59),(.56,.69),(.52,.78),(.43,.79),(.33,.72)]]),
dict(source='coco_10.jpg', title='摩托车黄色车身', mode='gold', reason='选择后车身黄色面板，排除骑手和车牌。', polygons=[
[(.60,.46),(.67,.44),(.69,.47),(.75,.50),(.82,.57),(.85,.64),(.82,.69),(.87,.77),(.86,.83),(.80,.89),(.69,.91),(.58,.91),(.48,.87),(.44,.81),(.48,.73),(.54,.70),(.57,.61)]],
exclude=[[(.72,.71),(.79,.71),(.79,.79),(.72,.79)]]),
dict(source='eg1.png', title='苹果表面', mode='fruit', reason='沿苹果外形选择果皮，不编辑白色背景和投影。', polygons=[
[(.22,.34),(.26,.30),(.33,.265),(.40,.25),(.43,.275),(.49,.24),(.57,.217),(.64,.213),(.72,.25),(.79,.32),(.825,.42),(.828,.53),(.80,.65),(.755,.73),(.68,.775),(.57,.79),(.45,.78),(.35,.755),(.28,.69),(.235,.59),(.216,.48)]]),
dict(source='eg2.png', title='旗帜布面', reason='沿旗面选择布料，排除旗杆和天空。', polygons=[
[(.282,.18),(.361,.258),(.465,.310),(.542,.365),(.611,.396),(.694,.398),(.764,.448),(.800,.452),(.800,.495),(.772,.558),(.758,.580),(.790,.625),(.795,.690),(.782,.724),(.762,.753),(.690,.776),(.556,.800),(.507,.803),(.444,.796),(.382,.784),(.319,.786),(.280,.782)]]),
dict(source='eg3.jpg', title='栈桥两侧湖水', reason='选择湖面，保留栈桥、森林、山体和天空。', polygons=[
[(0,.55),(.20,.54),(.40,.55),(.44,.57),(.40,.65),(.32,.80),(.15,.99),(0,.99)],
[(.57,.58),(.80,.56),(1,.57),(1,.99),(.85,.99),(.75,.83),(.66,.70),(.61,.62)]]),
dict(source='eg4.png', title='流程图内彩色图形', mode='colored', reason='仅选择下半部流程图内的彩色块与连线，不编辑上方正文。', polygons=[
[(.12,.53),(.89,.53),(.89,.92),(.12,.92)]]),
dict(source='eg5.png', title='人物背后的山体', reason='选择远处山坡，避开人物、脸部和前景装备。', polygons=[
[(0,.10),(.14,.11),(.26,.18),(.34,.25),(.35,.30),(.30,.31),(.26,.26),(.18,.31),(.08,.30),(0,.33)],
[(.56,.20),(.65,.12),(.80,.04),(.94,0),(1,0),(1,.28),(.88,.27),(.80,.25),(.73,.24),(.67,.28),(.60,.27)]]),
dict(source='eg6.png', title='走势图的彩色填充区', mode='colored', reason='选择图表中原有的红绿填充和蓝色曲线，保留黑色标题、坐标文字与白色页面。', polygons=[
[(.18,.24),(.90,.24),(.90,.74),(.18,.74)]]),
dict(source='eg7.png', title='流程图内彩色图形', mode='colored', reason='仅选择下半部流程图内的彩色块与连线，不编辑上方正文。', polygons=[
[(.12,.53),(.89,.53),(.89,.92),(.12,.92)]])]


def draw_polygon(draw, points, size, fill):
    w, h = size
    draw.polygon([(round(x*(w-1)), round(y*(h-1))) for x, y in points], fill=fill)


def make_mask(spec, image):
    if spec.get('mode') == 'approved_radio':
        return Image.open(ROOT / 'output/radio_body_region/radio_body_mask.png').convert('L').resize(image.size, Image.Resampling.NEAREST)
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in spec['polygons']:
        draw_polygon(draw, poly, image.size, 255)
    for poly in spec.get('exclude', []):
        draw_polygon(draw, poly, image.size, 0)
    for x1, y1, x2, y2 in spec.get('exclude_ellipses', []):
        w, h = image.size
        draw.ellipse((round(x1*w), round(y1*h), round(x2*w), round(y2*h)), fill=0)
    if spec.get('mode') in ('gold', 'colored', 'light_fur', 'fruit'):
        rgb = np.asarray(image).astype(np.int16)
        if spec['mode'] == 'gold':
            keep = (rgb[:,:,0] - rgb[:,:,2] > 8) & (rgb[:,:,1] - rgb[:,:,2] > 3) & (rgb.max(2)-rgb.min(2) > 20)
        elif spec['mode'] == 'light_fur':
            keep = (rgb.min(2) > 95) & (rgb.max(2)-rgb.min(2) < 75)
        elif spec['mode'] == 'fruit':
            keep = (rgb[:,:,0]-rgb[:,:,1] > 15) & (rgb[:,:,0]-rgb[:,:,2] > 25)
        else:
            keep = (rgb.max(2)-rgb.min(2) > 14) & (rgb.max(2) > 80)
        mask = Image.fromarray(np.where((np.asarray(mask) > 0) & keep, 255, 0).astype('uint8'))
    return mask


def main():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    for folder in ['masks_native', 'masks_model128', 'previews', 'detail_sheets']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    records, previews = [], []
    for spec in SPECS:
        path = SOURCE / spec['source']
        before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            image = source.convert('RGB')
        name = Path(spec['source']).stem
        mask = make_mask(spec, image)
        mask.save(OUT / 'masks_native' / f'{name}.png')
        h, w = smart_resize(image.height, image.width, factor=28,
                            min_pixels=64*28*28, max_pixels=128*28*28)
        model_mask = mask.resize((w, h), Image.Resampling.NEAREST)
        if spec['source'] == '0.png':
            model_mask = Image.open(ROOT / 'output/radio_body_region/radio_body_mask.png').convert('L')
            assert model_mask.size == (w, h)
        a = np.asarray(model_mask)
        assert set(np.unique(a)).issubset({0, 255}) and 0 < (a > 0).sum() < w*h
        model_mask.save(OUT / 'masks_model128' / f'{name}.png')
        # Generate an annotation preview only; source pixels are never overwritten.
        size = ImageOps.contain(image, (1000, 1000)).size
        base_image = image.resize(size, Image.Resampling.LANCZOS)
        preview_mask = mask.resize(size, Image.Resampling.NEAREST)
        tinted = Image.blend(base_image, Image.new('RGB', size, (0, 220, 110)), .40)
        preview = Image.composite(tinted, base_image, preview_mask)
        preview.save(OUT / 'previews' / f'{name}.jpg', quality=94)
        assert before_hash == hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict(source=spec['source'], source_path=str(path), source_sha256=before_hash,
            title=spec['title'], reason=spec['reason'], source_size=list(image.size), model_size=[w,h],
            native_mask=f'masks_native/{name}.png', model_mask=f'masks_model128/{name}.png',
            preview=f'previews/{name}.jpg', model_editable_pixels=int((a>0).sum()),
            model_editable_fraction=float((a>0).mean()), patch_box=[0,0,w,h],
            source_unchanged=True, training_started=False,
            note='Conservative manual proposal; small colored regions may offer limited optimization capacity.' if (a>0).mean()<.05 else 'Manual scene region; effect remains untested.')
        records.append(record)
        previews.append(preview)
        print(f'{spec["source"]}: {spec["title"]}, {100*record["model_editable_fraction"]:.1f}% at {w}x{h}', flush=True)
    (OUT / 'region_specs.json').write_text(json.dumps(SPECS, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    overview = Image.new('RGB', (1600, 6*350), '#f2f2f2')
    for i, (record, preview) in enumerate(zip(records, previews)):
        x, y = (i%4)*400, (i//4)*350
        thumb = ImageOps.contain(preview, (384, 284))
        overview.paste(thumb, (x+(400-thumb.width)//2, y+(284-thumb.height)//2))
        draw = ImageDraw.Draw(overview)
        draw.text((x+8,y+288), f'{record["source"]}  {record["title"]}', fill='black', font=small)
        draw.text((x+8,y+316), f'可编辑 {record["model_editable_fraction"]:.1%}', fill='#333333', font=small)
    overview.save(OUT / 'overview.jpg', quality=94)
    for start in range(0,len(records),6):
        sheet = Image.new('RGB',(1000,1290),'#f2f2f2')
        draw = ImageDraw.Draw(sheet)
        for j in range(min(6,len(records)-start)):
            record, preview = records[start+j], previews[start+j]
            x,y=(j%2)*500,(j//2)*430
            thumb=ImageOps.contain(preview,(480,370))
            sheet.paste(thumb,(x+(500-thumb.width)//2,y+(370-thumb.height)//2))
            draw.text((x+10,y+375),record['source']+' '+record['title'],font=font,fill='black')
            draw.text((x+10,y+405),f'绿色=可编辑；面积 {record["model_editable_fraction"]:.1%}',font=small,fill='#333333')
        sheet.save(OUT/'detail_sheets'/f'page_{start//6+1}.jpg',quality=95)
    lines=['# 场景内扰动区域候选', '', '本批 22 张图片仅选区域，未训练。绿色只是预览标注，不会添加到训练图片。', '',
           '收音机沿用已确认范围。其他区域是人工候选，不是自动精确分割；面积不强行统一，不能据此保证效果。', '',
           'masks_native 对应原图；masks_model128 按现有处理器 64–128 视觉 token 设置对齐。更改视觉预算后需要重新生成模型尺寸掩码。', '',
           '后续沿用现有自定义 mask-image 接口即可，掩码白色可改，黑色保持原图；本目录不提供自动训练队列。', '',
           '| 图片 | 选择区域 | 模型尺寸下占比 | 预览 |', '|---|---|---:|---|']
    for record in records:
        lines.append(f'| {record["source"]} | {record["title"]} | {record["model_editable_fraction"]:.1%} | [查看]({record["preview"]}) |')
    lines.extend(['', '文档和图表的可编辑区域较小，是为了避开正文、坐标文字及白色页面；效果尚未验证。', '',
                  '完整坐标、理由、文件哈希和模型尺寸见 region_specs.json 与 manifest.json。所有原图哈希检查通过。'])
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=[]
    for r in records:
        cards.append(f'<article><h2>{html.escape(r["source"])} — {html.escape(r["title"])}</h2><a href="{r["preview"]}"><img src="{r["preview"]}"></a><p>{html.escape(r["reason"])} 面积 {r["model_editable_fraction"]:.1%}</p></article>')
    (OUT/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>场景内扰动区域预览</title><style>body{font:16px sans-serif;background:#eee;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:white;padding:16px;border-radius:12px}h2{font-size:18px}img{width:100%;height:330px;object-fit:contain}p{line-height:1.6}</style><h1>场景内扰动区域候选 · 22 张</h1><p>绿色只标示可编辑范围，原图未改变。本次未启动任何训练。</p><main>'+''.join(cards)+'</main>',encoding='utf-8')


if __name__ == '__main__':
    main()
