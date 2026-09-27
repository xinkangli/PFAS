"""Check input alignment, grouped partitions, identities and molecular features."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/src'))
from featurize import Featurizer
from audit_data import identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--rebuild',action='store_true');a=p.parse_args()
    data=ROOT/'data'
    d=pd.read_csv(data/'final_observations.csv')
    assert d.record_id.is_unique
    admitted=d[d.included]
    assert len(admitted)==216 and admitted.compound_id.nunique()==19 and admitted.study_id.nunique()==6
    assert admitted.endpoint_type.value_counts().to_dict()=={'BAF':147,'BCF':69}
    for filename,expected in [('final_jobs.json',21),('similarity_jobs.json',7)]:
        jobs=json.loads((data/filename).read_text());assert len(jobs)==expected
        for j in jobs:
            tr=d[d.record_id.isin(j['train_ids'])];te=d[d.record_id.isin(j['test_ids'])]
            assert len(tr)==len(j['train_ids']) and len(te)==len(j['test_ids'])
            assert tr.included.all() and te.included.all()
            assert set(tr.record_id).isdisjoint(te.record_id)
            if j['design']!='study':
                assert set(tr.compound_id).isdisjoint(te.compound_id)
                assert set(te.compound_id)<=set(j['excluded_compounds'])
            else: assert set(tr.study_id).isdisjoint(te.study_id)
    tables={'bio':d,'silver':pd.read_csv(data/'silver_training.csv',low_memory=False),
            'gold':pd.read_csv(data/'auxiliary_wide.csv')}
    stored=np.load(data/'features_graph_corrected.npz');fz=Featurizer();rebuilt={}
    for key,table in tables.items():
        assert stored[key].shape==(len(table),1057)
        indices=np.arange(len(table)) if a.rebuild else np.unique(np.linspace(0,len(table)-1,min(32,len(table)),dtype=int))
        x,ok=fz.transform(table.iloc[indices].SMILES.tolist());assert ok.all()
        np.testing.assert_allclose(x,stored[key][indices],rtol=1e-6,atol=1e-5,equal_nan=True)
        ids=[identity(s)[2] for s in table.iloc[indices].SMILES]
        assert ids==table.iloc[indices].compound_id.tolist()
        if a.rebuild:rebuilt[key]=x
        print('PASS',key,len(table),'rows',len(indices),'regenerated feature rows',flush=True)
    if a.rebuild:
        out=ROOT/'outputs/rebuilt_features';out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'features_graph_corrected.npz',**rebuilt)
        (out/'feature_names.json').write_text(json.dumps(fz.feature_names_,indent=2))
        print('Rebuilt features match the supplied arrays:',out)
    # Package integrity excludes generated outputs and this manifest itself.
    manifest=ROOT/'data/SHA256.json'
    if manifest.exists():
        for name,digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print('PASS input data, admission, 28 fixed tasks, identities and feature alignment')

if __name__=='__main__':main()
