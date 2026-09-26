from pathlib import Path
import os,sys,hashlib
from functools import lru_cache
ROOT=Path(os.environ['NC_ROOT']);RUN=Path(os.environ['NC_RUN_DIR']);TRAIN_PYTHON=sys.executable
@lru_cache(maxsize=1)
def input_signature():
 h=hashlib.sha256()
 for p in sorted((RUN/'src').glob('*.py'))+[RUN/'data/silver_training.csv',RUN/'data/auxiliary_wide.csv',RUN/'data/features_graph_corrected.npz',RUN/'data/observations.csv',RUN/'data/fixed_splits.csv',RUN/'protocol.json']:
  h.update(p.name.encode());h.update(p.read_bytes())
 return h.hexdigest()
