"""One entry point for validation, an executable example, and complete training."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parent
STAGES = ['run_final.py', 'run_similarity.py', 'external.py', 'external_cv.py',
          'decision.py', 'summarize.py', 'paired_joint.py', 'validate_checkpoints.py']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'features', 'example', 'all'])
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'run')
    p.add_argument('--resume', action='store_true', help='Continue an existing training directory')
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    a = p.parse_args()
    env = dict(os.environ)
    if a.device: env['PFAS_DEVICE'] = a.device
    if a.action in ['check', 'features']:
        cmd = [sys.executable, str(ROOT/'code/check_data.py')]
        if a.action == 'features': cmd += ['--rebuild']
        subprocess.run(cmd, env=env, check=True)
        return
    dest = a.output.resolve()
    if dest == ROOT or ROOT in dest.parents and dest.parts[len(ROOT.parts)] != 'outputs':
        p.error('Choose outputs/<name> or a new directory outside this checkout.')
    if dest.exists() and not a.resume:
        p.error('Output exists. Choose a new directory or use --resume deliberately.')
    if not dest.exists():
        dest.mkdir(parents=True)
        shutil.copytree(ROOT/'data', dest/'data')
        shutil.copytree(ROOT/'code/src', dest/'src')
        for name in STAGES + ['protocol.json']:
            shutil.copy2(ROOT/'code'/name, dest/name)
        for name in ['checkpoints', 'results_final', 'results_similarity', 'audit']:
            (dest/name).mkdir()
    for name in ['data', 'src', 'protocol.json']:
        if not (dest/name).exists(): p.error('Incomplete run directory: '+name)
    stages = STAGES[:1] if a.action == 'example' else STAGES
    for name in stages:
        cmd = [sys.executable, str(dest/name)]
        if name in STAGES[:2]: cmd += ['--workers', '1']
        if a.action == 'example': cmd += ['--limit', '1']
        subprocess.run(cmd, cwd=dest, env=env, check=True)
    if a.action == 'example':
        subprocess.run([sys.executable, str(ROOT/'code/check_example.py'), str(dest)], env=env, check=True)
    print('Outputs:', dest)

if __name__ == '__main__': main()
