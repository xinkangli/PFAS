from pathlib import Path
import json,sys,os,hashlib
import numpy as np,pandas as pd,torch
from sklearn.ensemble import ExtraTreesRegressor
R=Path(__file__).resolve().parent;NC=R
os.environ['NC_ROOT']=str(NC);os.environ['NC_RUN_DIR']=str(R);sys.path[:0]=[str(R/'src')]
import train
from train import transport,downstream,scale,scale_fit,predict_mc
from model import MultiTaskTransportNet
from train_model1 import ENDPOINTS
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
from featurize import Featurizer
from audit_data import identity
O=R/'external_run'
for n in ['', 'checkpoints','results']:(O/n).mkdir(exist_ok=True)
train.OUT=O
d=pd.read_csv(R/'data/final_observations.csv');sv=pd.read_csv(R/'data/silver_training.csv',low_memory=False);gd=pd.read_csv(R/'data/auxiliary_wide.csv');raw0=np.load(R/'data/features_graph_corrected.npz')
e=pd.read_csv(R/'data/external_panel.csv');assert e.record_id.is_unique
assert e.compound_id.tolist()==[identity(str(s))[2] for s in e.SMILES]
assert set(e.source_doi).isdisjoint(set(d.doi_normalized))
from rdkit import Chem
fz=Featurizer();x,ok=fz.transform(e.SMILES.tolist());assert ok.all()
for i,s in enumerate(e.SMILES):
 m=Chem.MolFromSmiles(s);x[i,-5:]=[sum(a.GetAtomicNum()==9 for a in m.GetAtoms())]+[sum(a.GetAtomicNum()==6 and sum(n.GetAtomicNum()==9 for n in a.GetNeighbors())==k for a in m.GetAtoms()) for k in range(1,5)]
all_d=pd.concat([d,e],ignore_index=True);sx=np.r_[raw0['bio'],x];raw={'silver':raw0['silver'],'gold':raw0['gold'],'bio':sx};te=np.r_[np.zeros(len(d),bool),np.ones(len(e),bool)]
e.to_csv(O/'panel.csv',index=False)
protocol={'status':'fixed retrospective evaluation','interpretation':'retrospective sources already examined in previous development; not an untouched prospective test','modes':['source_transfer','strict_novel_chemical'],'primary_training':'216 source-qualified records','seeds':[42,43,44],'test_panel_union_exclusion':'Novel-in-bio subset removes all its IDs from both upstream stages; previously seen chemicals evaluated separately using compound-held-out folds','endpoints':'Source-designated endpoint and tissue retained; Sims short-term aqueous ratio is not regulatory steady-state BCF','panel_sha256':hashlib.sha256((O/'panel.csv').read_bytes()).hexdigest()}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
for seed in [42,43,44]:
 for mode in protocol['modes']:
  out=O/'results'/f'{mode}_{seed}.csv'
  if out.exists():continue
  mask=d.included.to_numpy(copy=True)
  evaluation_ids=set(e.compound_id) if mode=='source_transfer' else set(e.compound_id)-set(d.loc[mask,'compound_id'])
  tr=np.r_[mask,np.zeros(len(e),bool)];excluded=set() if mode=='source_transfer' else evaluation_ids
  print('START_EXTERNAL',mode,seed,'train',int(mask.sum()),d.loc[mask,'compound_id'].nunique(),flush=True)
  trans,key=transport(sv,gd,raw,excluded,seed);bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(all_d.loc[tr]).transform(all_d);rows=[]
  for model,xx in [('full_multitask',np.c_[sx,trans,bc]),('structure_context',np.c_[sx,np.zeros_like(trans),bc]),('extra_trees',np.c_[sx,bc])]:
   if model!='extra_trees':pred=downstream(xx,all_d,tr,te,seed,O/'checkpoints'/f'{mode}_{seed}_{model}.pt')
   else:
    pred=np.full((len(e),2),np.nan)
    for j,ep in enumerate(['BAF','BCF']):
     a=tr&all_d.endpoint_type.eq(ep).to_numpy()
     if all_d.loc[a,'compound_id'].nunique()<3:continue
     sc=scale_fit(xx[a]);est=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=.5,n_jobs=4,random_state=seed).fit(scale(xx[a],sc),all_d.loc[a,'log10_value_L_kg']);pred[:,j]=est.predict(scale(xx[te],sc))
   for i,v in e.iterrows():
    if v.compound_id not in evaluation_ids:continue
    j=['BAF','BCF'].index(v.endpoint_type)
    if d.loc[mask&d.endpoint_type.eq(v.endpoint_type).to_numpy(),'compound_id'].nunique()<3:continue
    rows.append({**v.to_dict(),'mode':mode,'seed':seed,'model':model,'predicted':float(pred[i,j]),'transport_key':key,'training_compounds':d.loc[mask,'compound_id'].nunique(),'test_seen_bio':v.compound_id in set(d.loc[mask,'compound_id'])})
  pd.DataFrame(rows).to_csv(out,index=False)
  (O/'results'/f'{mode}_{seed}_audit.json').write_text(json.dumps({'mode':mode,'seed':seed,'train_ids':d.loc[mask,'record_id'].tolist(),'excluded_compounds':sorted(excluded),'bio_chemical_overlap':len(set(d.loc[mask,'compound_id'])&set(e.compound_id)),'source_overlap':0,'transport_key':key},indent=2))
  print('DONE_EXTERNAL',mode,seed,len(rows),flush=True)

