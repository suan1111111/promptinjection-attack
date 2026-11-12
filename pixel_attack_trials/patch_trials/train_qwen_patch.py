"""Train one patch on frozen Qwen: random placements, validation-selected checkpoint.

Method: image_hijacks.components.processor.PatchImageProcessor. One patch tensor is
optimised; for every forward a placement is drawn uniformly over the whole placeable
range (``--position-sampling full``, the paper's behaviour); each update is projected
back into [0, 1].

Random placements make one forward a single sample of the loss expected over placements,
so the update rule and the number of samples per update matter:

  * a one-sample **sign** step is close to a random walk under random placements
    (measured: probe loss *rose* 0.017 over 12 steps), while SGD/Adam on the averaged
    gradient descend (measured: -0.05 / -0.08 over the same 12 steps);
  * therefore ``--questions-per-step`` questions, each with its own random placement,
    are averaged into one update (``--positions-per-step`` forwards per update);
  * a fixed-placement ``probe_loss`` (mean over fixed questions x fixed placements) and
    a validation split separate real progress from random-position noise.

Data split is never mixed: only ``split=train`` rows update the patch, ``split=val``
selects the checkpoint, ``split=test`` is only evaluated, after the selected weights
have already been written to disk. Nothing outside this file is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import optimize_qwen_pixels as base
from PIL import Image

KEYS = ['input_ids', 'attention_mask', 'labels', 'image_grid_thw']
PATCH_STRIDE = 28  # 14 px ViT patch, 2x2 merge -> one visual token per 28x28 px


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['fixed', 'random'], default='fixed')
    p.add_argument('--image', type=Path, default=Path(
        r'D:\CodexProjects\PrivaSee\data\datasets\meta_instruction\Soft-Prompts-Go-Hard-main\clean_images\0.png'
    ))
    p.add_argument('--csv', type=Path, default=Path(
        r'D:\CodexProjects\PrivaSee\data\datasets\meta_instruction\Soft-Prompts-Go-Hard-main\instruction_data\0\Attack\dataset.csv'
    ))
    p.add_argument('--model-dir', type=Path, default=base.DEFAULT_MODEL)
    p.add_argument('--output-dir', type=Path, default=HERE / 'outputs')
    p.add_argument('--patch-width', type=int, default=56)
    p.add_argument('--patch-height', type=int, default=56)
    p.add_argument('--x', type=int, default=None, help='Fixed/preview left coordinate; default right edge')
    p.add_argument('--y', type=int, default=0, help='Fixed/preview top coordinate')
    p.add_argument('--init', choices=['auto', 'crop', 'random'], default='auto')
    p.add_argument('--init-patch', type=Path, default=None,
                   help='Start from this saved patch PNG (continue a previous run)')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--questions-per-step', type=int, default=1)
    p.add_argument('--positions-per-step', type=int, default=4,
                   help='Placements (and forwards) averaged per update; random mode only')
    p.add_argument('--placement-fraction', type=float, default=1.0,
                   help='Restrict random placements to this fraction of the legal range, '
                        'centred on the preview point (1.0 = whole image)')
    p.add_argument('--position-sampling', choices=['full', 'content'], default='full',
                   help='full = uniform over the whole placeable range (paper behaviour). '
                        'content = skip constant regions, opt-in heuristic only')
    p.add_argument('--momentum', type=float, default=0.,
                   help='EMA factor on the sign/SGD gradient (0 disables; Adam ignores it)')
    p.add_argument('--probe-every', type=int, default=0,
                   help='Log the fixed-placement probe loss every N steps (0 disables)')
    p.add_argument('--probe-positions', type=int, default=4)
    p.add_argument('--probe-questions', type=int, default=4,
                   help='Fixed training questions averaged into one probe number')
    p.add_argument('--val-every', type=int, default=100,
                   help='Validation-loss checkpoint selection every N steps (0 disables)')
    p.add_argument('--val-positions', type=int, default=2,
                   help='Fixed placements averaged for each validation question')
    p.add_argument('--gen-check-every', type=int, default=0,
                   help='Generate with the current patch every N steps, for early stopping (0 disables)')
    p.add_argument('--gen-check-questions', type=int, default=2)
    p.add_argument('--gen-check-max-tokens', type=int, default=48)
    p.add_argument('--max-train-rows', type=int, default=5, help='0 uses all train rows')
    p.add_argument('--target-mode', choices=['short', 'full'], default='short')
    p.add_argument('--optimizer', choices=['auto', 'sign', 'sgd', 'adam'], default='auto',
                   help='auto = sign for a fixed placement, adam for random placements')
    p.add_argument('--alpha', type=float, default=1., help='Sign step size in 0-255 units')
    p.add_argument('--lr', type=float, default=None,
                   help='SGD learning rate (default 3) or Adam learning rate (default 0.02)')
    p.add_argument('--clip-grad', type=float, default=20., help='Gradient L2 limit for sgd/adam')
    p.add_argument('--max-visual-tokens', type=int, default=256)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--eval-rows', type=int, default=0,
                   help='split=test rows evaluated after training; 0 disables generation')
    p.add_argument('--eval-positions', type=int, default=0,
                   help='Placements per test row for evaluation_positions.json')
    p.add_argument('--eval-max-new-tokens', type=int, default=128)
    p.add_argument('--validate-only', action='store_true', help='Check inputs and geometry, no model weights')
    p.add_argument('--gradient-check-only', action='store_true', help='One real backward/update step only')
    return p.parse_args()


def select_target(row, mode):
    full_target = (row.get('target_answer') or '').strip()
    if mode != 'short':
        return full_target
    short = (row.get('short_target') or '').strip()
    if not short:
        short = base.SHORT_TARGETS.get((row.get('category') or '').strip(), '')
    return short or full_target


def read_split_rows(path, split, mode, limit=0, require_target=True):
    """Rows of one CSV split, never mixing splits."""
    selected = []
    for line, row in enumerate(base.read_dataset_rows(path), 2):
        if (row.get('split') or '').strip().lower() != split:
            continue
        question = (row.get('question') or '').strip()
        target = select_target(row, mode)
        if not question or (require_target and not target):
            raise ValueError(f'CSV line {line}: missing question or target for split={split}')
        selected.append(dict(csv_line=int(row.get('source_csv_line', line)), question=question,
                             training_target=target,
                             normal_answer=(row.get('normal_answer') or '').strip(),
                             target_answer=(row.get('target_answer') or '').strip(),
                             marker=(row.get('target_contains') or '').strip(),
                             category=(row.get('category') or '').strip()))
    if not selected:
        raise ValueError(f'No split={split} rows')
    return selected[:limit] if limit else selected


def read_training_rows(path, mode, limit):
    return read_split_rows(path, 'train', mode, limit)


def compose(clean, patch, x, y):
    """CopySlices preserves patch gradients and leaves all other pixels unchanged."""
    h, w = patch.shape[-2:]
    H, W = clean.shape[-2:]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError('Patch outside image')
    image = clean.clone()
    image[:, :, y:y+h, x:x+w] = patch
    return image


def outside_difference(clean, image, x, y, w, h):
    diff = (image.detach() - clean).abs().clone()
    diff[:, :, y:y+h, x:x+w] = 0
    return float(diff.max().item())


def content_bounds(clean, torch, threshold=0.01):
    """Opt-in heuristic for --position-sampling content: non-constant rows/columns.

    Standard deviation only; it marks constant regions, and a uniform but genuine image
    region would also be skipped, which is why the default stays 'full'.
    """
    luminance = clean.detach().abs().mean(dim=1)[0]
    columns = torch.nonzero(luminance.std(dim=0) > threshold).flatten()
    rows = torch.nonzero(luminance.std(dim=1) > threshold).flatten()
    if columns.numel() == 0 or rows.numel() == 0:
        return None
    return int(columns[0]), int(columns[-1]), int(rows[0]), int(rows[-1])


def placement_range(bounds, W, H, w, h):
    """Inclusive (x0, x1, y0, y1) of allowed patch top-left coordinates."""
    full = (0, W - w, 0, H - h)
    if bounds is None:
        return full
    x0, x1, y0, y1 = bounds
    x0, x1 = max(0, min(x0, W - w)), max(0, min(x1 - w + 1, W - w))
    y0, y1 = max(0, min(y0, H - h)), max(0, min(y1 - h + 1, H - h))
    if x1 < x0 or y1 < y0:
        return full
    return x0, x1, y0, y1


def draw_placement(rng, x0, x1, y0, y1):
    return rng.randrange(x0, x1 + 1), rng.randrange(y0, y1 + 1)


def tensor_to_pil(pixels):
    array = (
        pixels.detach().clamp(0, 1).mul(255).round().byte().cpu()
        .squeeze(0).permute(1, 2, 0).numpy()
    )
    return Image.fromarray(array, mode='RGB')


def normalize_text(text):
    return re.sub(r'\s+', '', (text or '').strip().lower())


def contains(haystack, needle):
    needle = normalize_text(needle)
    return bool(needle) and needle in normalize_text(haystack)


def is_degenerate(text):
    """Cheap flag for empty / repeated-token output; a heuristic, not a verdict."""
    norm = normalize_text(text)
    if len(norm) < 2:
        return True
    return len(norm) > 12 and len(set(norm)) <= 2


def classify_answer(answer, target, normal, marker):
    """Per-answer facts only; semantic quality still needs human review."""
    target_hit = contains(answer, target)
    normal_hit = contains(answer, normal) if normal else False
    marker_hit = bool(marker) and normalize_text(marker) in normalize_text(answer)
    if is_degenerate(answer):
        kind = 'degenerate'
    elif target_hit and normal_hit:
        kind = 'target_and_normal'
    elif target_hit:
        kind = 'target_only'
    elif normal_hit:
        kind = 'normal_only'
    else:
        kind = 'other'
    return dict(target_full_match=target_hit, normal_match=normal_hit,
                marker_match=marker_hit, kind=kind)


def summarize(records):
    answers = [ans for record in records for ans in record['answers']]
    total = len(answers)
    kinds = {}
    for ans in answers:
        kinds[ans['kind']] = kinds.get(ans['kind'], 0) + 1
    return {
        'answers': total,
        'target_full_match': sum(ans['target_full_match'] for ans in answers),
        'marker_match': sum(ans['marker_match'] for ans in answers),
        'normal_preserved': sum(ans['normal_match'] for ans in answers),
        'kinds': kinds,
        'questions_whose_clean_answer_already_matches_target':
            sum(record['clean_target_full_match'] for record in records),
    }


def generate_answer(processor, model, image, question, max_new_tokens, torch):
    """Greedy answer for one 8-bit PIL image (same path as a real deployment)."""
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                             {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, use_cache=True,
                                   max_new_tokens=max_new_tokens)
    answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    del inputs, generated
    return answer


def build_held_out(rows, model, processor, run, clean, patch, placements, max_new_tokens, torch):
    """split=test generations with the selected patch pasted at several placements.

    Uses 8-bit pixels, exactly like the saved PNG, because that is what would be pasted
    onto a real image. The clean-image answer is generated once per question so the
    "already contained the target before any attack" case is recorded explicitly.
    """
    records = []
    for row in rows:
        question, target, normal, marker = (row['question'], row['target_answer'],
                                            row['normal_answer'], row['marker'])
        record = {'csv_line': row['csv_line'], 'question': question, 'target_answer': target,
                  'normal_answer': normal, 'target_marker': marker,
                  'clean_answer': '', 'clean_target_full_match': False,
                  'clean_normal_match': False, 'clean_kind': '', 'answers': []}
        record['clean_answer'] = generate_answer(processor, model, tensor_to_pil(clean),
                                                 question, max_new_tokens, torch)
        clean_facts = classify_answer(record['clean_answer'], target, normal, marker)
        record['clean_target_full_match'] = clean_facts['target_full_match']
        record['clean_normal_match'] = clean_facts['normal_match']
        record['clean_kind'] = clean_facts['kind']
        for px, py in placements:
            image = tensor_to_pil(compose(clean, patch, px, py))
            answer = generate_answer(processor, model, image, question, max_new_tokens, torch)
            facts = classify_answer(answer, target, normal, marker)
            record['answers'].append({'x': int(px), 'y': int(py), 'answer': answer, **facts})
            del image
            base.write_json(run / 'evaluation_positions.json', {
                'split': 'test', 'positions': [list(p) for p in placements],
                'records': records + [record], 'complete': False,
                'notice': 'Patch pasted at several held-out placements; '
                          'literal matches are not a semantic judgement.',
            })
        records.append(record)
    payload = {'split': 'test', 'positions': [list(p) for p in placements],
               'records': records, 'summary': summarize(records), 'complete': True,
               'notice': 'Patch pasted at several held-out placements; '
                         'literal matches are not a semantic judgement.'}
    base.write_json(run / 'evaluation_positions.json', payload)
    return payload


def main():
    a = parse_args()
    if a.questions_per_step < 1 or a.positions_per_step < 1 or a.probe_positions < 1:
        raise ValueError('questions/positions per step must be positive')
    for name in ['steps', 'patch_width', 'patch_height', 'save_every']:
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    if a.max_train_rows < 0 or a.max_visual_tokens < 64:
        raise ValueError('Invalid row/token limit')
    if a.eval_rows < 0 or a.eval_positions < 0 or a.eval_max_new_tokens < 1:
        raise ValueError('Invalid evaluation limits')
    if a.probe_every < 0 or a.val_every < 0 or a.probe_questions < 1 or a.val_positions < 1:
        raise ValueError('Invalid probe/validation settings')
    if not 0 <= a.momentum < 1:
        raise ValueError('--momentum must be in [0, 1)')
    if not all(math.isfinite(v) and v > 0 for v in [a.alpha, a.clip_grad] + ([a.lr] if a.lr else [])):
        raise ValueError('Optimizer parameters must be positive and finite')
    if a.alpha > 255:
        raise ValueError('alpha must be <=255')
    a.image, a.csv, a.model_dir = [v.expanduser().resolve() for v in [a.image, a.csv, a.model_dir]]
    if not (a.model_dir / 'model.safetensors.index.json').is_file():
        raise ValueError('Local model directory missing weight index')
    rows = read_training_rows(a.csv, a.target_mode, a.max_train_rows)
    val_every = 0 if a.gradient_check_only else a.val_every
    val_rows = []
    if val_every:
        try:
            val_rows = read_split_rows(a.csv, 'val', a.target_mode)
        except ValueError:
            base.status('No split=val rows in this CSV: checkpoint selection falls back to the last step.')
            val_every = 0
    test_rows = []
    if a.eval_rows or a.eval_positions:
        test_rows = read_split_rows(a.csv, 'test', a.target_mode)
        test_rows = test_rows[:a.eval_rows] if a.eval_rows else test_rows
    with Image.open(a.image) as src:
        clean_pil = src.convert('RGB')
    base.status(f'{a.mode}: {len(rows)} train / {len(val_rows)} val / {len(test_rows)} test rows; '
                'only train rows update the patch. Importing torch...')
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(a.seed)
    order_rng, loc_rng = random.Random(a.seed), random.Random(a.seed + 1)
    fixed_rng = random.Random(a.seed + 2)  # probe/eval placements only
    processor = AutoProcessor.from_pretrained(a.model_dir, min_pixels=64*28*28,
        max_pixels=a.max_visual_tokens*28*28, use_fast=False, local_files_only=True)
    def make_examples(source):
        return [base.prepare_example(processor, clean_pil, r['question'], r['training_target'], torch)
                for r in source]
    examples = make_examples(rows)
    val_examples = make_examples(val_rows) if val_rows else []
    ref = examples[0]
    for group, name in ((examples, 'train'), (val_examples, 'val')):
        if any(not torch.equal(e['image_grid_thw'], ref['image_grid_thw']) for e in group):
            raise RuntimeError(f'Inconsistent visual grids in split={name}')
    mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)
    clean = base.unpatchify_to_pixels(ref['official_pixel_values'], ref['image_grid_thw'][0], mean, std, torch)
    error = float((base.patchify_pixels(clean, mean, std) - ref['official_pixel_values']).abs().max())
    if error > 1e-5:
        raise RuntimeError(f'Preprocessing mismatch: {error}')
    H, W = clean.shape[-2:]
    w, h = a.patch_width, a.patch_height
    if w > W or h > H:
        raise ValueError(f'Patch {w}x{h} does not fit the {W}x{H} model input')
    bounds = content_bounds(clean, torch) if a.position_sampling == 'content' else None
    x0, x1, y0, y1 = placement_range(bounds, W, H, w, h)
    if a.x is None:
        x = x1 if a.mode == 'fixed' else x0 + (x1 - x0) // 2
    else:
        x = a.x
    y = a.y if a.mode == 'fixed' or a.x is not None else y0 + (y1 - y0) // 2
    if a.mode == 'random' and 0 < a.placement_fraction < 1:
        if not 0 < a.placement_fraction:
            raise ValueError('--placement-fraction must be in (0, 1]')
        half_w = int((x1 - x0) * a.placement_fraction / 2)
        half_h = int((y1 - y0) * a.placement_fraction / 2)
        x0, x1 = max(0, x - half_w), min(W - w, x + half_w)
        y0, y1 = max(0, y - half_h), min(H - h, y + half_h)
    compose(clean, torch.zeros(1, 3, h, w), x, y)
    init = ('crop' if a.mode == 'fixed' else 'random') if a.init == 'auto' else a.init
    preview_std = float(clean[:, :, y:y+h, x:x+w].std())
    tokens = max(1, (h // PATCH_STRIDE) * (w // PATCH_STRIDE))
    base.status(f'Model input {W}x{H}; patch {w}x{h} ({w*h/(W*H):.2%} of the model input, '
                f'~{tokens} visual tokens); placements x[{x0},{x1}] y[{y0},{y1}] '
                f'({(x1-x0+1)*(y1-y0+1)} legal); sampling={a.position_sampling}; '
                f'preview ({x},{y}); init={init}')
    if preview_std < 0.01:
        base.status('Warning: the preview placement is constant, not image content; pass --x/--y.')
    # Generation must see the same visual grid as training. The processor re-derives the
    # grid from whatever image it is handed, and its min_pixels can upscale an image that
    # is already at the training resolution (e.g. 252x168 -> 280x196), which silently
    # turns a converged patch into a no-op at evaluation time.
    eval_processor = AutoProcessor.from_pretrained(a.model_dir, min_pixels=4*28*28,
        max_pixels=max(W * H, a.max_visual_tokens * 28 * 28), use_fast=False,
        local_files_only=True)
    probe_grid = eval_processor.image_processor(
        images=[tensor_to_pil(clean)], return_tensors='pt')['image_grid_thw'][0].tolist()
    expected_grid = [1, H // 14, W // 14]
    if probe_grid != expected_grid:
        raise RuntimeError(f'Evaluation preprocessing mismatch: grid {probe_grid} != '
                           f'training grid {expected_grid}')
    base.status(f'Evaluation grid check passed: {probe_grid} (same as training)')
    if a.validate_only:
        base.status('Input, split isolation, labels, preprocessing and geometry checks passed; '
                    'no model weights loaded.')
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support required')
    base.status('Loading local Qwen NF4 model...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model_dir,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map={'': 0}, low_cpu_mem_usage=True, local_files_only=True, torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    device = next(model.parameters()).device
    clean, mean, std = [v.to(device) for v in [clean, mean, std]]
    gen = torch.Generator(device=device).manual_seed(a.seed)
    initial = clean[:, :, y:y+h, x:x+w].clone() if init == 'crop' else torch.rand((1,3,h,w), device=device, generator=gen)
    if a.init_patch is not None:
        source = a.init_patch.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as handle:
            loaded = torch.tensor(np.array(handle.convert('RGB'))).permute(2, 0, 1)[None]
        loaded = loaded.float().div(255).to(device)
        if loaded.shape[-2:] != (h, w):
            loaded = torch.nn.functional.interpolate(loaded, size=(h, w), mode='bilinear',
                                                     align_corners=False)
        initial = loaded
        init = f'patch:{source.name}'
        base.status(f'Initialising the patch from {source}')
    patch = initial.detach().clone().requires_grad_(True)
    # Random placements give a high-variance gradient; plain SGD/Adam keep the coherent
    # part, a sign step would amplify the noise, so 'auto' picks sign only when fixed.
    optimizer_name = a.optimizer if a.optimizer != 'auto' else ('sign' if a.mode == 'fixed' else 'adam')
    lr = a.lr if a.lr is not None else (0.02 if optimizer_name == 'adam' else 3.)
    torch_optimizer = (torch.optim.Adam([patch], lr=lr) if optimizer_name == 'adam' else
                       torch.optim.SGD([patch], lr=lr) if optimizer_name == 'sgd' else None)
    use_ema = a.momentum > 0 and optimizer_name != 'adam'
    momentum = torch.zeros_like(patch) if use_ema else None
    base.status(f'Update rule: {optimizer_name}' + (f' (lr={lr:g})' if torch_optimizer else
                f' (alpha={a.alpha:g}/255)') + (f', gradient EMA {a.momentum:g}' if use_ema else '')
                + (f', clip_grad={a.clip_grad:g}' if torch_optimizer else ''))
    positions_per_step = 1 if a.mode == 'fixed' else a.positions_per_step
    probe_questions = list(range(min(a.probe_questions, len(rows))))
    if a.mode == 'fixed':
        # One legal placement: probes/validation/eval must use it, not random ones.
        probe_positions = val_positions = eval_placements = [(x, y)]
    else:
        probe_positions = [draw_placement(fixed_rng, x0, x1, y0, y1) for _ in range(a.probe_positions)]
        val_positions = [draw_placement(fixed_rng, x0, x1, y0, y1) for _ in range(a.val_positions)]
        eval_placements = [draw_placement(fixed_rng, x0, x1, y0, y1) for _ in range(a.eval_positions)]
    gen_check_rows = rows[:min(a.gen_check_questions, len(rows))]
    gen_checks = []
    run = a.output_dir.expanduser().resolve() / f'{a.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True, exist_ok=False)
    steps = 1 if a.gradient_check_only else a.steps
    cfg = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    cfg.update(training_size=[W,H], actual_steps=steps, actual_init=init, preview_xy=[x,y],
        train_rows=len(rows), val_rows=len(val_rows), test_rows=len(test_rows),
        preprocessing_max_error=error, evaluation_performed=False,
        actual_positions_per_step=positions_per_step, placement_range=[x0,x1,y0,y1],
        resolved_optimizer=optimizer_name, resolved_lr=lr if torch_optimizer else None,
        gradient_ema=use_ema, clip_grad_applied=bool(torch_optimizer) and a.clip_grad > 0,
        validation_enabled=bool(val_every), val_every_effective=val_every,
        content_bounds=list(bounds) if bounds else None,
        training_grid=[1, H // 14, W // 14], evaluation_grid=probe_grid,
        probe_questions=[rows[i]['csv_line'] for i in probe_questions],
        probe_positions=[list(p) for p in probe_positions],
        val_positions=[list(p) for p in val_positions],
        eval_positions=[list(p) for p in eval_placements],
        image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest())
    base.write_json(run/'config.json', cfg)
    base.write_json(run/'training_rows.json', {'train': rows, 'val': val_rows,
                                             'test': test_rows})
    base.save_tensor_image(run/'clean_resized.png', clean)
    base.save_tensor_image(run/'patch_initial.png', initial)
    base.save_tensor_image(run/'initial_preview.png', compose(clean, initial, x, y))
    order = list(range(len(rows)))
    torch.cuda.reset_peak_memory_stats(device)

    def example_kwargs(e):
        return {k: e[k].to(device) for k in KEYS}

    def eval_at(question_examples, question_indices, placements):
        """Mean loss over fixed (question, placement) pairs; no gradients."""
        total = 0.
        with torch.no_grad():
            for qi in question_indices:
                for px, py in placements:
                    image = compose(clean, patch, px, py)
                    pixel_values = base.patchify_pixels(image, mean, std)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = model(**example_kwargs(question_examples[qi]),
                                    pixel_values=pixel_values, use_cache=False, return_dict=True)
                    total += float(out.loss.detach())
                    del out, image, pixel_values
        return total / (len(question_indices) * len(placements))

    base.status(f'Training {steps} steps, {positions_per_step} forward(s) per update. Output: {run}')
    baseline_probe = eval_at(examples, probe_questions, probe_positions) if a.probe_every else None
    baseline_val = eval_at(val_examples, list(range(len(val_examples))), val_positions) if val_every else None
    selected = {'step': 0, 'val_loss': baseline_val, 'patch': initial.detach().clone()}
    val_trace = []
    if baseline_probe is not None:
        base.status(f'Baseline probe loss (before any update) = {baseline_probe:.6f}')
    if baseline_val is not None:
        base.status(f'Baseline validation loss (before any update) = {baseline_val:.6f}')
        val_trace.append({'step': 0, 'val_loss': baseline_val})
    with (run/'training_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['step','row_index','csv_line','x','y','loss_before_update','grad_mean','grad_max',
                  'outside_max_diff','allocated_gib','peak_allocated_gib','probe_loss','val_loss']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if baseline_probe is not None or baseline_val is not None:
            writer.writerow({'step': 0, 'probe_loss': baseline_probe if baseline_probe is not None else '',
                             'val_loss': baseline_val if baseline_val is not None else ''})
            f.flush()
        for step in range(1, steps+1):
            indices = order_rng.sample(order, min(a.questions_per_step, len(order)))
            if a.mode == 'fixed':
                placements = [(idx, x, y) for idx in indices]  # one forward per question
            else:
                placements = [(indices[slot % len(indices)],) + draw_placement(loc_rng, x0, x1, y0, y1)
                              for slot in range(positions_per_step)]
            patch.grad = None
            loss_value = 0.
            share = 1. / len(placements)
            for idx, px, py in placements:
                image = compose(clean, patch, px, py)
                pixel_values = base.patchify_pixels(image, mean, std)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**example_kwargs(examples[idx]), pixel_values=pixel_values,
                        use_cache=False, return_dict=True)
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss_value += float(loss.detach()) * share
                (loss * share).backward()
                del output, loss, image, pixel_values
            g = patch.grad
            if g is None or not torch.isfinite(g).all() or float(g.abs().max()) == 0:
                raise RuntimeError('Missing, nonfinite or zero patch gradient')
            gm, gx = float(g.abs().mean()), float(g.abs().max())
            with torch.no_grad():
                direction = g
                if momentum is not None:
                    momentum.mul_(a.momentum).add_(g)
                    direction = momentum
                if torch_optimizer is not None:
                    patch.grad = direction
                    if a.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_([patch], a.clip_grad)
                    torch_optimizer.step()
                    patch.grad = None
                else:
                    patch.add_(direction.sign(), alpha=-a.alpha/255)
                patch.clamp_(0,1)
                if not torch.isfinite(patch).all():
                    raise RuntimeError('Nonfinite patch after update')
                px, py = placements[-1][1], placements[-1][2]
                updated = compose(clean, patch, px, py)
                outside = outside_difference(clean, updated, px, py, w, h)
                if outside != 0:
                    raise RuntimeError('Pixels outside patch changed')
            probe = ''
            if a.probe_every and (step == steps or step % a.probe_every == 0):
                probe = f'{eval_at(examples, probe_questions, probe_positions):.6f}'
            val = ''
            if val_every and (step == steps or step % val_every == 0):
                val_loss = eval_at(val_examples, list(range(len(val_examples))), val_positions)
                val = f'{val_loss:.6f}'
                val_trace.append({'step': step, 'val_loss': val_loss})
                improved = selected['val_loss'] is None or val_loss < selected['val_loss']
                if improved:
                    selected = {'step': step, 'val_loss': val_loss, 'patch': patch.detach().clone()}
                base.status(f'validation loss {val_loss:.6f} at step {step}'
                            + (' (new best)' if improved else ''))
            writer.writerow({
                'step': step,
                'row_index': json.dumps([p[0] for p in placements]),
                'csv_line': json.dumps([rows[p[0]]['csv_line'] for p in placements]),
                'x': json.dumps([p[1] for p in placements]),
                'y': json.dumps([p[2] for p in placements]),
                'loss_before_update': f'{loss_value:.8f}',
                'grad_mean': f'{gm:.10g}',
                'grad_max': f'{gx:.10g}',
                'outside_max_diff': outside,
                'allocated_gib': f'{torch.cuda.memory_allocated(device)/1024**3:.4f}',
                'peak_allocated_gib': f'{torch.cuda.max_memory_allocated(device)/1024**3:.4f}',
                'probe_loss': probe,
                'val_loss': val,
            })
            f.flush()
            if step == 1 or step == steps or step % a.save_every == 0:
                base.save_tensor_image(run/f'patch_step_{step:04d}.png', patch)
                base.save_tensor_image(run/f'full_step_{step:04d}.png', updated)
                base.status(f'{step}/{steps} loss={loss_value:.5f} grad={gm:.3e} '
                            f'placements={[(p[1], p[2]) for p in placements]}'
                            + (f' probe={probe}' if probe else ''))
            if a.gen_check_every and (step == steps or step % a.gen_check_every == 0):
                fresh = []
                for slot, row in enumerate(gen_check_rows):
                    _, px, py = placements[slot % len(placements)]
                    image = tensor_to_pil(compose(clean, patch, px, py))
                    answer = generate_answer(eval_processor, model, image, row['question'],
                                             a.gen_check_max_tokens, torch)
                    fresh.append({'step': step, 'csv_line': row['csv_line'],
                                  'x': int(px), 'y': int(py),
                                  'target_match': contains(answer, row['target_answer']),
                                  'answer': answer})
                    del image
                gen_checks.extend(fresh)
                base.write_json(run/'generation_checks.json',
                                {'checks': gen_checks,
                                 'matches': sum(c['target_match'] for c in gen_checks)})
                base.status(f'generation check @{step}: '
                            f'{sum(c["target_match"] for c in fresh)}/{len(fresh)} '
                            f'answers contain the target'
                            + (f' | {fresh[0]["answer"][:110]!r}' if fresh else ''))
            del updated, g
    if selected['val_loss'] is None:  # no validation run: the last step is the result
        selected = {'step': steps, 'val_loss': None, 'patch': patch.detach()}
    base.write_json(run/'validation_trace.json', {'selection': 'lowest mean val loss over '
                     'val questions x fixed val placements', 'trace': val_trace,
                     'selected_step': selected['step'], 'selected_val_loss': selected['val_loss']})
    selected_patch = selected['patch'] if selected['val_loss'] is not None else patch.detach()
    # Weights first: generation can fail or run out of memory after this point.
    torch.save({'patch': selected_patch.detach().cpu(), 'selected_step': selected['step'],
                'selected_val_loss': selected['val_loss'], 'config': cfg, 'completed_steps': steps},
               run/'patch_final.pt')
    base.save_tensor_image(run/'patch_final.png', selected_patch)
    base.save_tensor_image(run/'patch_last_step.png', patch)
    base.save_tensor_image(run/'full_final_preview.png', compose(clean, selected_patch, x, y))
    base.save_tensor_image(run/'adversarial_final.png', compose(clean, selected_patch, x, y))
    if probe_positions:
        tiles = [compose(clean, selected_patch, px, py) for px, py in probe_positions[:4]]
        if len(tiles) == 4:
            grid = torch.cat([torch.cat(tiles[:2], dim=3), torch.cat(tiles[2:], dim=3)], dim=2)
        else:
            grid = torch.cat(tiles, dim=3)
        base.save_tensor_image(run/'patch_placements_grid.png', grid)
    evaluated_positions = None
    if a.eval_positions and test_rows and not a.gradient_check_only:
        evaluated_positions = build_held_out(test_rows, model, eval_processor, run, clean,
                                             selected_patch, eval_placements,
                                             a.eval_max_new_tokens, torch)
    evaluated = base.evaluate_saved_images(a, model, eval_processor, run, torch) is not None
    cfg['evaluation_performed'] = evaluated
    cfg['evaluation_positions_performed'] = evaluated_positions is not None
    cfg['selected_step'] = selected['step']
    cfg['selected_val_loss'] = selected['val_loss']
    base.write_json(run/'config.json', cfg)
    base.write_json(run/'result.json', dict(completed_steps=steps, gradient_check_only=a.gradient_check_only,
        evaluation_performed=evaluated, evaluation_positions_performed=evaluated_positions is not None,
        selection='validation loss, split=val', selected_step=selected['step'],
        selected_val_loss=selected['val_loss'], baseline_probe=baseline_probe,
        baseline_val=baseline_val, final_loss_before_update=loss_value, last_grad_max=gx,
        positions_per_step=positions_per_step, probe_loss=probe,
        generation_checks=dict(count=len(gen_checks),
                               matches=sum(c['target_match'] for c in gen_checks)),
        evaluation_summary=(evaluated_positions or {}).get('summary'),
        outside_max_diff=outside, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        preview_xy=[x,y], patch_file=str(run/'patch_final.png')))
    base.status(f'Completed: {run}. Selected step {selected["step"]}. '
                f'Held-out evaluation performed: {evaluated}.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f'Failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
