from pathlib import Path
import os, sys, hashlib
from functools import lru_cache
ROOT = Path(os.environ['NC_ROOT'])
RUN = Path(os.environ['NC_RUN_DIR'])
TRAIN_PYTHON = sys.executable

@lru_cache(maxsize=1)
def input_signature():
    h = hashlib.sha256()
    files = sorted((RUN / 'src').glob('*.py'))
    files += [RUN / 'data' / n for n in [
        'silver_training.csv', 'auxiliary_wide.csv',
        'features_graph_corrected.npz', 'final_observations.csv',
        'final_jobs.json', 'similarity_jobs.json']]
    files += [RUN / 'protocol.json']
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()
