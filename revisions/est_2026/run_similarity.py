from pathlib import Path
import os,sys,json,hashlib,argparse,time
import numpy as np,pandas as pd,torch
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
R=Path(__file__).resolve().parent
os.environ['NC_ROOT']=str(R);os.environ['NC_RUN_DIR']=str(R)
sys.path.insert(0,str(R/'src'))
import train
from train import transport,downstream,scale_fit,scale
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK

for folder in ['checkpoints','results_final','results_similarity','audit']:(R/folder).mkdir(exist_ok=True)

def run():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 sv=pd.read_csv(R/'data/silver_training.csv',low_memory=False);gd=pd.read_csv(R/'data/auxiliary_wide.csv');d=pd.read_csv(R/'data/final_observations.csv');raw=np.load(R/'data/features_graph_corrected.npz');sx=raw['bio']
 jobs=json.loads((R/'data/similarity_jobs.json').read_text());count=0
 fps={c:AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,nBits=1024) for c,s in d[['compound_id','group_smiles']].drop_duplicates().itertuples(index=False,name=None)}
 for ji,job in enumerate(jobs):
  if ji%args.workers!=args.worker:continue
  name=job['task'];result=R/'results_similarity'/(name+'.csv');done=R/'results_similarity'/(name+'_audit.json')
  if result.exists() and done.exists():continue
  tr=d.record_id.isin(job['train_ids']).to_numpy();te=d.record_id.isin(job['test_ids']).to_numpy();seed=job['seed'];excluded=set(job['excluded_compounds'])
  assert not (tr&te).any()
  print('START',name,time.strftime('%H:%M:%S'),flush=True)
  trans,key=transport(sv,gd,raw,excluded,seed)
  shuffled,skey=transport(sv,gd,raw,excluded,seed,True)
  bc=BioEncoder(BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK).fit(d.loc[tr]).transform(d)
  full=np.c_[sx,trans,bc]
  variants=[('full_multitask',full,False,key),('structure_context',np.c_[sx,np.zeros_like(trans),bc],False,key),('transport_context',np.c_[np.zeros_like(sx),trans,bc],False,key),('full_single_task',full,True,key),('shuffled_auxiliary_labels',np.c_[sx,shuffled,bc],False,skey)]
  rows=[];skips=[]
  def append(model,ix,pred,tkey=''):
   for i,v in zip(ix,pred):
    r=d.iloc[i]
    rows.append({'task':name,'design':job['design'],'cohort':job['cohort'],'seed':seed,'model':model,'record_id':r.record_id,'compound_id':r.compound_id,'study_id':r.study_id,'endpoint':r.endpoint_type,'observed':r.log10_value_L_kg,'predicted':float(v),'transport_key':tkey,'test_chemical_seen_in_bio_train':r.compound_id in set(d.loc[tr,'compound_id'])})
  eligible={ep:d.loc[tr&d.endpoint_type.eq(ep).to_numpy(),'compound_id'].nunique()>=3 for ep in ['BAF','BCF']}
  for model,x,single,tkey in variants:
   pred=downstream(x,d,tr,te,seed,R/'checkpoints'/(name+'_'+model+'.pt'),single)
   for ei,ep in enumerate(['BAF','BCF']):
    if not eligible[ep]:continue
    mask=d.loc[te,'endpoint_type'].eq(ep).to_numpy();ix=np.flatnonzero(te)[mask]
    append(model,ix,pred[mask,ei],tkey)
  for ep in ['BAF','BCF']:
   a=tr&d.endpoint_type.eq(ep).to_numpy();b=te&d.endpoint_type.eq(ep).to_numpy()
   if not b.any():continue
   if not eligible[ep]:skips.append({'endpoint':ep,'test_records':int(b.sum()),'reason':'fewer than 3 training compounds'});continue
   y=d.loc[a,'log10_value_L_kg'].to_numpy();ix=np.flatnonzero(b)
   simple=np.c_[sx[:,[0,1,3,12,13]],sx[:,1038:1052],bc]
   models=[('simple_graph_context',simple,Ridge(alpha=100)),('extra_trees_structure_context',np.c_[sx,bc],ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_features=.5,n_jobs=4,random_state=seed)),('hist_gradient_structure_context',np.c_[sx,bc],HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=1,random_state=seed))]
   for model,x,est in models:
    sc=scale_fit(x[a]);est.fit(scale(x[a],sc),y);append(model,ix,est.predict(scale(x[b],sc)))
   cx=scale(bc,scale_fit(bc[a]));trainids=d.loc[a,'compound_id'].unique();pred=[]
   for i in ix:
    sim={c:DataStructs.TanimotoSimilarity(fps[c],fps[d.iloc[i].compound_id]) for c in trainids};mx=max(sim.values());candidates=a&d.compound_id.isin([c for c,v in sim.items() if abs(v-mx)<1e-12]).to_numpy();inds=np.flatnonzero(candidates);distance=((cx[inds]-cx[i])**2).sum(1);chosen=inds[np.lexsort((d.loc[candidates,'record_id'].to_numpy(),distance))[0]];pred.append(d.iloc[chosen].log10_value_L_kg)
   append('nearest_chemical_context',ix,pred)
  out=pd.DataFrame(rows,columns=['task','design','cohort','seed','model','record_id','compound_id','study_id','endpoint','observed','predicted','transport_key','test_chemical_seen_in_bio_train']);assert np.isfinite(out.predicted.to_numpy(dtype=float)).all()
  assert not out.duplicated(['model','record_id']).any()
  out.to_csv(result,index=False)
  meta={**job,'models':sorted(out.model.unique()),'skipped':skips,'chemical_intersection':len(set(d.loc[tr,'compound_id'])&set(d.loc[te,'compound_id'])),'study_intersection':len(set(d.loc[tr,'study_id'])&set(d.loc[te,'study_id'])),'upstream_keys':[key,skey],'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'final_data_sha256':hashlib.sha256((R/'data/final_observations.csv').read_bytes()).hexdigest(),'completed':time.strftime('%Y-%m-%d %H:%M:%S')}
  done.write_text(json.dumps(meta,indent=2));print('DONE',name,len(rows),flush=True);count+=1
  if args.limit and count>=args.limit:break
 print('WORKER_COMPLETE',args.worker,flush=True)
if __name__=='__main__':run()
