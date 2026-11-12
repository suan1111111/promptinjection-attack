"""Wait for the already-running smoke queue, then launch the full serial batch.

No model jobs overlap. Any failed smoke job blocks the full batch.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / 'outputs/seven_attack_revised_smoke_20261027/status.json'
WORK = ROOT / 'outputs/seven_attack_revised_20261027'


def main():
    WORK.mkdir(exist_ok=True)
    state = WORK / 'driver_status.json'
    def report(status, **extra):
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=status, **extra), indent=2), encoding='utf-8')
        temporary.replace(state)
        print(status, extra, flush=True)
    report('waiting_for_smoke', pid=os.getpid())
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        try:
            states = json.loads(SMOKE.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            states = {}
        if any(v.get('state') == 'failed' for v in states.values()):
            report('blocked_by_smoke_failure')
            return 1
        if len(states) == 7 and all(v.get('state') == 'completed' for v in states.values()):
            break
        time.sleep(10)
    else:
        report('smoke_timeout')
        return 1
    command = [sys.executable, str(ROOT/'pixel_attack_trials/run_seven_previews.py'),
               '--revised', '--steps', '1000', '--work-dir', str(WORK)]
    report('training', pid=os.getpid(), command=command)
    with (WORK/'batch.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1'),
                                stdout=log, stderr=subprocess.STDOUT)
    report('completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
