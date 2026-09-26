"""Create a clean, independent training folder without precomputed predictions."""
from pathlib import Path
import argparse,shutil
p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args()
root=Path(__file__).resolve().parent;dest=Path(a.destination).resolve()
if dest.exists():raise SystemExit('Destination must not exist; use a new directory.')
dest.mkdir(parents=True)
for n in ['data','src']:shutil.copytree(root/n,dest/n)
for n in ['run_final.py','run_similarity.py','external.py','external_cv.py','decision.py','summarize.py','paired_joint.py','validate_checkpoints.py','protocol.json']:shutil.copy2(root/n,dest/n)
for n in ['checkpoints','results_final','results_similarity','audit']:(dest/n).mkdir()
print(dest)
