"""Validate actual fitted predictions and reload the neural model checkpoints."""
from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
import torch
R=Path(sys.argv[1]).resolve()
os.environ['NC_ROOT']=str(R);os.environ['NC_RUN_DIR']=str(R);sys.path.insert(0,str(R/'src'))
from train import DEVICE,scale,tensor,predict_mc
from model import MultiTaskTransportNet
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
job=json.loads((R/'data/final_jobs.json').read_text())[0];task=job['task']
meta=json.loads((R/'results_final'/(task+'_audit.json')).read_text())
pred=pd.read_csv(R/'results_final'/(task+'.csv'))
assert len(pred)==324 and pred.model.nunique()==9 and np.isfinite(pred.predicted).all()
assert not pred.duplicated(['model','record_id']).any()
assert meta['chemical_intersection']==0
d=pd.read_csv(R/'data/final_observations.csv');sx=np.load(R/'data/features_graph_corrected.npz')['bio']
tr=d.record_id.isin(job['train_ids']).to_numpy();te=d.record_id.isin(job['test_ids']).to_numpy()
bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(d.loc[tr]).transform(d)
checks=[]
for model in ['full_multitask','structure_context','transport_context','full_single_task','shuffled_auxiliary_labels']:
    key=meta['upstream_keys'][int(model=='shuffled_auxiliary_labels')]
    info=json.loads((R/'checkpoints'/('transport_'+key+'.json')).read_text())
    assert info['silver_intersection']==info['auxiliary_intersection']==0
    assert info['pretrain_epochs']==40 and info['finetune_epochs']==400
    assert set(job['excluded_compounds'])<=set(info['excluded_compounds'])
    tx=np.load(R/'checkpoints'/('transport_'+key+'.npz'))['bio']
    xx=np.c_[np.zeros_like(sx) if model=='transport_context' else sx,np.zeros_like(tx) if model=='structure_context' else tx,bc]
    ck=torch.load(R/'checkpoints'/(task+'_'+model+'.pt'),map_location='cpu')
    values=np.full((te.sum(),2),np.nan)
    for z in ck['models']:
        net=MultiTaskTransportNet(in_dim=xx.shape[1],endpoints=[['BAF','BCF'][j] for j in z['targets']],hidden=(512,256,128),dropout=.25).to(DEVICE)
        net.load_state_dict(z['state_dict']);net.eval()
        with torch.no_grad():values[:,z['targets']]=net(tensor(scale(xx[te],ck['input_scaler']))).cpu().numpy()*z['target_sd']+z['target_mean']
    dd=d.loc[te,['record_id','endpoint_type']].copy()
    dd['recomputed']=[values[i,0 if ep=='BAF' else 1] for i,ep in enumerate(dd.endpoint_type)]
    joined=pred[pred.model.eq(model)].merge(dd,on='record_id',validate='one_to_one')
    delta=float(abs(joined.predicted-joined.recomputed).max());assert delta<1e-5
    checks.append({'model':model,'max_reload_error':delta})
# Stochastic dropout must not update BatchNorm statistics.
bn=[m for m in net.modules() if isinstance(m,torch.nn.BatchNorm1d)]
before=[m.running_mean.clone() for m in bn]
predict_mc(net,scale(xx[te],ck['input_scaler']),z['target_mean'],z['target_sd'],n=3)
assert all(torch.equal(v,m.running_mean) for v,m in zip(before,bn))
(R/'audit/example_validation.json').write_text(json.dumps({'predictions':len(pred),'models':9,'checkpoint_reload':checks,'batchnorm_unchanged':True},indent=2))
print('PASS fresh example: 9 models, 324 predictions, strict upstream exclusion, checkpoint reload and BatchNorm')
