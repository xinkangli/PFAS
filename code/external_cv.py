from pathlib import Path
import json,os,sys,hashlib
import pandas as pd,numpy as np,torch
from sklearn.ensemble import ExtraTreesRegressor
R=Path(__file__).resolve().parent;NC=R;os.environ['NC_ROOT']=str(NC);os.environ['NC_RUN_DIR']=str(R);sys.path.insert(0,str(R/'src'))
from train import DEVICE,scale,scale_fit,predict_mc,tensor
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
from train_model1 import ENDPOINTS
from model import MultiTaskTransportNet
from featurize import Featurizer
from rdkit import Chem
O=R/'external_run';d=pd.read_csv(R/'data/final_observations.csv');e=pd.read_csv(O/'panel.csv');raw=np.load(R/'data/features_graph_corrected.npz');x,ok=Featurizer().transform(e.SMILES.tolist());assert ok.all()
for i,s in enumerate(e.SMILES):
 m=Chem.MolFromSmiles(s);x[i,-5:]=[sum(a.GetAtomicNum()==9 for a in m.GetAtoms())]+[sum(a.GetAtomicNum()==6 and sum(n.GetAtomicNum()==9 for n in a.GetNeighbors())==k for a in m.GetAtoms()) for k in range(1,5)]
sx=np.r_[raw['bio'],x];all_d=pd.concat([d,e],ignore_index=True);rows=[];routes=[]
for seed in [42,43,44]:
 for fold in range(5):
  name=f'final_primary216_chemical_{seed}_{fold}';meta=json.loads((R/'results_final'/(name+'_audit.json')).read_text());tr=d.record_id.isin(meta['train_ids']).to_numpy();known_test=set(d.loc[d.record_id.isin(meta['test_ids']),'compound_id']);sel=e.compound_id.isin(known_test).to_numpy()
  if not sel.any():continue
  key=meta['upstream_keys'][0];info=json.loads((R/'checkpoints'/('transport_'+key+'.json')).read_text());assert known_test<=set(info['excluded_compounds'])
  state=torch.load(R/'checkpoints'/('transport_'+key+'.pt'),map_location='cpu');torch.manual_seed(seed)
  net=MultiTaskTransportNet(in_dim=sx.shape[1],endpoints=ENDPOINTS,hidden=(512,256,128),dropout=.3).to(DEVICE);net.load_state_dict(state['state_dict']);ext_trans=predict_mc(net,scale(x,state['input_scaler']),state['target_mean'],state['target_sd']);del net
  tc=np.r_[np.load(R/'checkpoints'/('transport_'+key+'.npz'))['bio'],ext_trans];bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(d.loc[tr]).transform(all_d)
  for model,xx in [('full_multitask',np.c_[sx,tc,bc]),('structure_context',np.c_[sx,np.zeros_like(tc),bc]),('extra_trees',np.c_[sx,bc])]:
   pred=np.full((len(e),2),np.nan)
   if model!='extra_trees':
    ck=torch.load(R/'checkpoints'/(name+'_'+model+'.pt'),map_location='cpu');xe=scale(xx[len(d):],ck['input_scaler'])
    for z in ck['models']:
     nn=MultiTaskTransportNet(in_dim=xx.shape[1],endpoints=[['BAF','BCF'][j] for j in z['targets']],hidden=(512,256,128),dropout=.25).to(DEVICE);nn.load_state_dict(z['state_dict']);nn.eval()
     with torch.no_grad():pred[:,z['targets']]=nn(tensor(xe)).cpu().numpy()*z['target_sd']+z['target_mean']
     del nn
   else:
    for j,ep in enumerate(['BAF','BCF']):
     a=tr&d.endpoint_type.eq(ep).to_numpy()
     if d.loc[a,'compound_id'].nunique()<3:continue
     sc=scale_fit(xx[:len(d)][a]);est=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=.5,n_jobs=4,random_state=seed).fit(scale(xx[:len(d)][a],sc),d.loc[a,'log10_value_L_kg']);pred[:,j]=est.predict(scale(xx[len(d):],sc))
   for i,v in e.loc[sel].iterrows():
    j=['BAF','BCF'].index(v.endpoint_type)
    if d.loc[tr&d.endpoint_type.eq(v.endpoint_type).to_numpy(),'compound_id'].nunique()<3:continue
    assert v.compound_id not in set(d.loc[tr,'compound_id'])
    rows.append({**v.to_dict(),'mode':'strict_chemical_out','seed':seed,'model':model,'predicted':float(pred[i,j]),'transport_key':key,'fold':fold,'test_seen_bio':False,'training_compounds':d.loc[tr,'compound_id'].nunique()})
  routes.append({'task':name,'external_record_ids':e.loc[sel,'record_id'].tolist(),'excluded_compounds':info['excluded_compounds'],'training_record_ids':meta['train_ids'],'upstream_key':key})
for p in sorted((O/'results').glob('strict_novel_chemical_*.csv')):
 a=pd.read_csv(p);a['mode']='strict_chemical_out';rows.extend(a.to_dict('records'))
pred=pd.DataFrame(rows);assert not pred.duplicated(['record_id','model','seed']).any();assert not pred.test_seen_bio.any()
pred.to_csv(O/'results/strict_crossfitted_predictions.csv',index=False);(O/'results/strict_routing_audit.json').write_text(json.dumps(routes,indent=2));print('STRICT_EXTERNAL_COMPLETE',len(pred),pred.record_id.nunique())
