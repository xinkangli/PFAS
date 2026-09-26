from pathlib import Path
import os,sys,json,numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parent;os.environ['NC_ROOT']=str(R);os.environ['NC_RUN_DIR']=str(R);sys.path.insert(0,str(R/'src'))
from train import DEVICE,scale,tensor
from model import MultiTaskTransportNet
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
D=pd.read_csv(R/'data/final_observations.csv');sx=np.load(R/'data/features_graph_corrected.npz')['bio'];out=[]
tasks=[('results_final','final_primary216_chemical_42_0'),('results_final','final_primary216_chemical_44_4'),('results_similarity','final_similarity70_0')]
for folder,task in tasks:
 a=json.loads((R/folder/(task+'_audit.json')).read_text());res=pd.read_csv(R/folder/(task+'.csv'));tr=D.record_id.isin(a['train_ids']).to_numpy();te=D.record_id.isin(a['test_ids']).to_numpy();bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(D.loc[tr]).transform(D)
 for model in ['full_multitask','structure_context','transport_context','full_single_task','shuffled_auxiliary_labels']:
  rr=res[res.model.eq(model)];key=a['upstream_keys'][1 if model=='shuffled_auxiliary_labels' else 0];tx=np.load(R/f'checkpoints/transport_{key}.npz')['bio'];xx=np.c_[np.zeros_like(sx) if model=='transport_context' else sx,np.zeros_like(tx) if model=='structure_context' else tx,bc];ck=torch.load(R/f'checkpoints/{task}_{model}.pt',map_location='cpu');pred=np.full((te.sum(),2),np.nan)
  for z in ck['models']:
   n=MultiTaskTransportNet(in_dim=xx.shape[1],endpoints=[['BAF','BCF'][j] for j in z['targets']],hidden=(512,256,128),dropout=.25).to(DEVICE);n.load_state_dict(z['state_dict']);n.eval()
   with torch.no_grad():pred[:,z['targets']]=n(tensor(scale(xx[te],ck['input_scaler']))).cpu().numpy()*z['target_sd']+z['target_mean']
   del n
  dd=D.loc[te,['record_id','endpoint_type']].copy();dd['recomputed']=[pred[i,0 if ep=='BAF' else 1] for i,ep in enumerate(dd.endpoint_type)];z=rr.merge(dd,on='record_id',validate='one_to_one');diff=float(abs(z.predicted-z.recomputed).max()) if len(z) else 0.;assert diff<1e-5,(task,model,diff);out.append({'task':task,'model':model,'predictions':len(z),'max_abs_difference':diff})
(R/'audit/checkpoint_reload_validation.json').write_text(json.dumps(out,indent=2));print('CHECKPOINT_RELOAD_PASSED',len(out),max(x['max_abs_difference'] for x in out))
