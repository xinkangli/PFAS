from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd,torch
from sklearn.metrics import mean_absolute_error
R=Path(__file__).resolve().parent;NC=R;os.environ['NC_ROOT']=str(NC);os.environ['NC_RUN_DIR']=str(R);sys.path.insert(0,str(R/'src'))
import train
from train import DEVICE,transport,downstream,scale,predict_mc
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
from model import MultiTaskTransportNet
O=R/'decision_run'
for n in ['', 'checkpoints','results']:(O/n).mkdir(exist_ok=True)
d=pd.read_csv(R/'data/final_observations.csv');raw=np.load(R/'data/features_graph_corrected.npz');sv=pd.read_csv(R/'data/silver_training.csv',low_memory=False);gd=pd.read_csv(R/'data/auxiliary_wide.csv');mask=d.v15_primary.to_numpy();sx=raw['bio']
def prediction_samples(path,x,n=30):
 c=torch.load(path,map_location='cpu');b=scale(x,c['input_scaler']);out=np.full((n,len(x),2),np.nan)
 for q in c['models']:
  net=MultiTaskTransportNet(in_dim=x.shape[1],endpoints=[['BAF','BCF'][j] for j in q['targets']],hidden=(512,256,128),dropout=.25).to(DEVICE);net.load_state_dict(q['state_dict']);net.eval()
  for module in net.modules():
   if isinstance(module,torch.nn.Dropout):module.train()
  with torch.no_grad():
   for k in range(n):out[k][:,q['targets']]=net(train.tensor(b)).cpu().numpy()*q['target_sd']+q['target_mean']
  del net
 return out
protocol={'calibration':'9 calibration compounds, up to 4 test compounds, remaining source-qualified compounds train; grouping not records','levels':[.5,.8,.9,.95],'nonvacuity':'finite-sample rank above available calibration groups yields infinite radius, retained explicitly','seeds':[42,43,44],'no_tuning_on_test':True}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
for seed in [42,43,44]:
 result=O/'results'/f'calibration_{seed}.csv'
 if result.exists():continue
 rng=np.random.default_rng(seed);ids=rng.permutation(np.sort(d.loc[mask,'compound_id'].unique()));testids=set(ids[:4]);calids=set(ids[4:13]);tr=mask&~d.compound_id.isin(testids|calids).to_numpy();ca=mask&d.compound_id.isin(calids).to_numpy();te=mask&d.compound_id.isin(testids).to_numpy()
 print('START_CALIBRATION',seed,flush=True);tc,key=transport(sv,gd,raw,testids|calids,seed);bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(d.loc[tr]).transform(d);x=np.c_[sx,tc,bc];members=[]
 for init in [seed*10+i for i in range(3)]:
  path=O/'checkpoints'/f'calibration_{seed}_{init}.pt';pred=downstream(x,d,tr,ca|te,init,path);members.append(prediction_samples(path,x,30))
 draws=np.concatenate(members);mean=np.nanmean(draws,0);std=np.nanstd(draws,0);rows=[];rawrows=[]
 for j,ep in enumerate(['BAF','BCF']):
  if d.loc[tr&d.endpoint_type.eq(ep).to_numpy(),'compound_id'].nunique()<3:continue
  c=ca&d.endpoint_type.eq(ep).to_numpy();t=te&d.endpoint_type.eq(ep).to_numpy()
  if not c.any() or not t.any():continue
  scales=np.maximum(std[:,j],.05);absres=np.abs(d.log10_value_L_kg.to_numpy()-mean[:,j]);scores=pd.DataFrame({'id':d.loc[c,'compound_id'],'score':absres[c]/scales[c]}).groupby('id').score.max().sort_values().to_numpy();n=len(scores)
  for level in [.5,.8,.9,.95]:
   rank=int(np.ceil((n+1)*level));q=scores[rank-1] if rank<=n else np.inf;covered=absres[t]<=q*scales[t];bundle=pd.DataFrame({'id':d.loc[t,'compound_id'],'covered':covered}).groupby('id').covered.all()
   rows.append({'seed':seed,'endpoint':ep,'nominal':level,'calibration_compounds':n,'test_compounds':len(bundle),'test_records':int(t.sum()),'rank':rank,'radius_multiplier':q,'record_coverage':float(covered.mean()),'chemical_bundle_coverage':float(bundle.mean()),'mean_width':float(np.mean(2*q*scales[t])),'finite':bool(np.isfinite(q))})
  for i in np.flatnonzero(t):rawrows.append({'seed':seed,'record_id':d.iloc[i].record_id,'compound_id':d.iloc[i].compound_id,'endpoint':ep,'observed':d.iloc[i].log10_value_L_kg,'predicted':mean[i,j],'uncertainty':std[i,j]})
 pd.DataFrame(rows).to_csv(result,index=False);pd.DataFrame(rawrows).to_csv(O/'results'/f'calibration_predictions_{seed}.csv',index=False)
 (O/'results'/f'calibration_{seed}_split.json').write_text(json.dumps({'train_ids':d.loc[tr,'record_id'].tolist(),'calibration_ids':d.loc[ca,'record_id'].tolist(),'test_ids':d.loc[te,'record_id'].tolist(),'transport_key':key,'excluded_compounds':sorted(testids|calids)},indent=2))
 print('DONE_CALIBRATION',seed,flush=True)

print('CALIBRATION_COMPLETE',flush=True)
