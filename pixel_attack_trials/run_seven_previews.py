"""Serial seven-form pilot. Final folder contains only composed attack PNGs.

All jobs use clean_images/0.png and its own 0/Attack/dataset.csv injection column.
Existing completed jobs are verified and skipped; failed pixel jobs resume.
Patch retries preserve earlier failed runs and start afresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

from attack_dataset import read_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def write_json(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--revised', action='store_true')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    final = work / 'final_images'
    final.mkdir(exist_ok=True)
    repo = ROOT / 'data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main'
    image = repo / 'clean_images/0.png'
    csv_path = repo / 'instruction_data/0/Attack/dataset.csv'
    rows = read_rows(csv_path)
    if any('<maliciousurl>' not in row['target_answer'] for row in rows):
        raise ValueError('All rows must share the same URL-injection marker')
    common = ['--image', str(image), '--csv', str(csv_path), '--target-mode', 'full',
        '--max-train-rows', '40', '--steps', str(args.steps), '--max-visual-tokens', '128',
        '--seed', '7', '--save-every', '50', '--alpha', '1', '--eval-rows', '5',
        '--eval-max-new-tokens', '128']
    specs = [
        ('01_global', 'global', ['--epsilon','16']),
        ('02_border', 'border', ['--epsilon','32','--border-width','8']),
        ('03_lowfreq', 'lowfreq', ['--epsilon','16','--frequency-ratio','0.125']),
        ('04_flower', 'flower', ['--epsilon','16','--patch-width','56','--patch-height','56']),
        ('05_sparse', 'sparse', ['--epsilon','32','--sparse-pixels','256']),
        ('06_fixed_patch', 'fixed', ['--init','random','--patch-width','56','--patch-height','56']),
        ('07_random_patch', 'random', ['--init','random','--patch-width','56','--patch-height','56']),
    ]
    if args.revised:
        common[common.index('--eval-rows')+1] = '10'
        common[common.index('--eval-max-new-tokens')+1] = '512'
        specs[3] = ('04_flower', 'flower', ['--epsilon','255','--patch-width','90','--patch-height','90'])
        specs[4] = ('05_sparse', 'sparse', ['--epsilon','32','--sparse-pixels','256','--alpha','4'])
    identity = dict(image=str(image), csv=str(csv_path), steps=args.steps, revised=args.revised,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(), common=common, specs=specs)
    manifest = work / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != json.loads(json.dumps(identity)):
        raise ValueError('Existing manifest differs; select a new work directory')
    write_json(manifest, identity)
    state_path = work / 'status.json'
    states = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    for name, mode, options in specs:
        previous = states.get(name, {})
        if previous.get('state') == 'completed' and (final / f'{name}.png').is_file():
            continue
        patch = mode in ('fixed','random')
        output = work / name
        output.mkdir(exist_ok=True)
        command = [sys.executable, str(HERE / ('patch_trials/train_qwen_patch.py' if patch else 'optimize_qwen_pixels.py'))] + common + options
        if patch:
            command += ['--mode',mode,'--output-dir',str(output)]
            if args.revised:
                command += ['--questions-per-step','4']
        else:
            run = output / 'run'
            if run.exists() and not (run / 'checkpoint.pt').exists():
                run.rename(output / ('incomplete_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            command += ['--constraint',mode,'--questions-per-step','4' if args.revised else '1','--run-dir',str(run)]
            if args.revised:
                command += ['--update-rule','revised']
            if (run / 'checkpoint.pt').exists():
                command += ['--resume',str(run / 'checkpoint.pt')]
        states[name] = dict(state='running', command=command, started=datetime.now().isoformat())
        write_json(state_path, states)
        print(f'START {name}', flush=True)
        with (output / 'process.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        states[name]['returncode'] = result.returncode
        states[name]['finished'] = datetime.now().isoformat()
        if result.returncode:
            states[name]['state'] = 'failed'
            write_json(state_path, states)
            print(f'FAILED {name}; see {output / "process.log"}', flush=True)
            continue
        if patch:
            candidates = sorted(output.glob('*/result.json'), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise RuntimeError(f'Missing patch result for {name}')
            run = candidates[-1].parent
        result_data = json.loads((run / 'result.json').read_text(encoding='utf-8'))
        if result_data['completed_steps'] != args.steps:
            raise RuntimeError(f'Unexpected update count for {name}')
        source = run / 'adversarial_final.png'
        from PIL import Image
        with Image.open(source) as saved:
            saved.verify()
        destination = final / f'{name}.png'
        shutil.copy2(source, destination)
        states[name].update(state='completed', run_dir=str(run), final_image=str(destination))
        write_json(state_path, states)
        print(f'COMPLETE {name}: {destination}', flush=True)
    successful = sum(item.get('state') == 'completed' for item in states.values())
    print(f'FINISHED {successful}/7; final images: {final}', flush=True)
    return 0 if successful == 7 else 1


if __name__ == '__main__':
    raise SystemExit(main())
