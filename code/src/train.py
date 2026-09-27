"""Strict, resumable transport pretraining -> auxiliary fitting -> biological fitting."""
from nc_config import ROOT as CONFIG_ROOT,RUN as CONFIG_RUN,TRAIN_PYTHON,input_signature

from pathlib import Path
import sys,json,hashlib,argparse,time
import numpy as np,pandas as pd,torch
ROOT=CONFIG_ROOT;OUT=CONFIG_RUN
import os
from model import MultiTaskTransportNet
from train_model1 import ENDPOINTS
from train_model2 import BioEncoder,BIO_CAT_COLS,BIO_NUM_COLS,BIO_TOPK
torch.set_num_threads(4)
DEVICE=os.environ.get('PFAS_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
def scale_fit(a):
 a=np.where(np.isfinite(a),a,np.nan);med=np.nanmedian(a,axis=0);med=np.nan_to_num(med)
 v=np.where(np.isnan(a),med,a);mu=v.mean(0);sd=v.std(0);sd[sd<1e-8]=1
 return med,mu,sd
def scale(a,s):return ((np.where(np.isfinite(a),a,s[0])-s[1])/s[2]).astype('float32')
def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=DEVICE)
def loss(p,y,m):
 n=m.sum(0);ok=n>0
 return (((p-y).square()*m).sum(0)[ok]/n[ok]).mean()
def predict_mc(net,x,mu,sd,n=30):
 net.eval()
 for module in net.modules():
  if isinstance(module,torch.nn.Dropout):module.train()
 vals=[]
 with torch.no_grad():
  for _ in range(n):vals.append(net(tensor(x)).cpu().numpy()*sd+mu)
 return np.concatenate([np.mean(vals,axis=0),np.std(vals,axis=0)],axis=1).astype('float32')
def _transport_unlocked(sv,gd,raw,excluded,seed,shuffled=False):
 key=hashlib.sha256(json.dumps([input_signature(),sorted(excluded),seed,shuffled],sort_keys=True).encode()).hexdigest()[:16]
 target=OUT/'checkpoints'/('transport_'+key+'.pt');predfile=target.with_suffix('.npz');audit=target.with_suffix('.json')
 if predfile.exists():return np.load(predfile)['bio'],key
 sm=~sv.compound_id.isin(excluded).to_numpy();gm=~gd.compound_id.isin(excluded).to_numpy()
 assert not set(sv.loc[sm,'compound_id'])&excluded
 assert not set(gd.loc[gm,'compound_id'])&excluded
 if gm.sum()<2:raise ValueError('Insufficient auxiliary chemicals after strict exclusion')
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 xs=raw['silver'][sm];xg=raw['gold'][gm];ys=sv.loc[sm,'label'].to_numpy(dtype=float)
 yg=gd.loc[gm,ENDPOINTS].to_numpy(dtype=float)
 permutations=[]
 if shuffled:
  # Silver labels shuffled across chemical groups (not independent duplicate rows).
  ids=sv.loc[sm,'compound_id'].to_numpy();u=np.unique(ids);mapping=dict(zip(u,rng.permutation(u)))
  groups={c:g.v.to_numpy() for c,g in pd.DataFrame({'id':ids,'v':ys}).groupby('id',sort=False)};ys=np.array([groups[mapping[c]][i%len(groups[mapping[c]])] for i,c in enumerate(ids)])
  for j in range(yg.shape[1]):
   ix=np.flatnonzero(np.isfinite(yg[:,j]));perm=rng.permutation(ix);yg[ix,j]=yg[perm,j];permutations.append(perm.tolist())
 scaler=scale_fit(np.concatenate([xs,xg]));xs=scale(xs,scaler);xg=scale(xg,scaler)
 mu=np.nanmean(yg,axis=0);sd=np.nanstd(yg,axis=0);k=ENDPOINTS.index('logKow')
 allkow=np.r_[ys,yg[:,k][np.isfinite(yg[:,k])]];mu[k]=allkow.mean();sd[k]=allkow.std()
 mu=np.nan_to_num(mu);sd=np.where(np.isfinite(sd)&(sd>1e-8),sd,1.)
 mask=np.isfinite(yg).astype('float32');yn=np.nan_to_num((yg-mu)/sd)
 net=MultiTaskTransportNet(in_dim=xs.shape[1],endpoints=ENDPOINTS,hidden=(512,256,128),dropout=.3).to(DEVICE)
 opt=torch.optim.Adam(net.parameters(),lr=.001);xt=tensor(xs);st=tensor((ys-mu[k])/sd[k])
 for epoch in range(40):
  net.train();perm=torch.randperm(len(xs),device=DEVICE)
  for ix in perm.split(512):
   err=(net(xt[ix])[:,k]-st[ix]).square().mean();opt.zero_grad();err.backward();opt.step()
 opt=torch.optim.Adam(net.parameters(),lr=.0003,weight_decay=.001);xt=tensor(xg);yt=tensor(yn);mt=tensor(mask)
 for epoch in range(400):
  net.train();err=loss(net(xt),yt,mt);opt.zero_grad();err.backward();opt.step()
 pred=predict_mc(net,scale(raw['bio'],scaler),mu,sd)
 state={'state_dict':net.cpu().state_dict(),'input_scaler':scaler,'target_mean':mu,'target_sd':sd,'endpoints':ENDPOINTS,'excluded_compounds':sorted(excluded),'seed':seed,'shuffled':shuffled}
 torch.save(state,target)
 np.savez_compressed(predfile,bio=pred)
 info={'key':key,'excluded_compounds':sorted(excluded),'silver_rows':int(sm.sum()),'silver_compounds':sv.loc[sm,'compound_id'].nunique(),'auxiliary_compounds':int(gm.sum()),'auxiliary_counts':dict(zip(ENDPOINTS,mask.sum(0).astype(int).tolist())),'silver_intersection':0,'auxiliary_intersection':0,'shuffled_labels_retrained':shuffled,'seed':seed,'pretrain_epochs':40,'finetune_epochs':400,'preprocess_fit_ids':'silver and auxiliary training IDs only','auxiliary_train_ids':gd.loc[gm,'compound_id'].tolist(),'auxiliary_permutation_indices':permutations}
 audit.write_text(json.dumps(info,indent=2));print('TRANSPORT_DONE',key,shuffled,int(sm.sum()),int(gm.sum()),flush=True)
 return pred,key
def transport(sv,gd,raw,excluded,seed,shuffled=False):
 import fcntl
 lockkey=hashlib.sha256(json.dumps([sorted(excluded),seed,shuffled],sort_keys=True).encode()).hexdigest()[:16]
 with (OUT/'checkpoints'/('transport_'+lockkey+'.lock')).open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  return _transport_unlocked(sv,gd,raw,excluded,seed,shuffled)

def downstream(x,d,tr,te,seed,path,single=False):
 scaler=scale_fit(x[tr]);a=scale(x[tr],scaler);b=scale(x[te],scaler)
 y=d.loc[tr,'log10_value_L_kg'].to_numpy();e=d.loc[tr,'endpoint_type'].map({'BAF':0,'BCF':1}).to_numpy();out=np.full((int(te.sum()),2),np.nan)
 saved=[]
 for targets in ([[0],[1]] if single else [[0,1]]):
  keep=np.isin(e,targets)
  targets=[j for j in targets if (e==j).sum()>0]
  if not targets:continue
  mu=np.array([y[e==j].mean() for j in targets]);sd=np.array([max(y[e==j].std(),.1) for j in targets])
  yy=np.zeros((keep.sum(),len(targets)),dtype='float32');m=np.zeros_like(yy)
  for k,j in enumerate(targets):m[e[keep]==j,k]=1;yy[e[keep]==j,k]=(y[keep][e[keep]==j]-mu[k])/sd[k]
  torch.manual_seed(seed);net=MultiTaskTransportNet(in_dim=x.shape[1],endpoints=[['BAF','BCF'][j] for j in targets],hidden=(512,256,128),dropout=.25).to(DEVICE)
  opt=torch.optim.Adam(net.parameters(),lr=.0005,weight_decay=.0001);xt=tensor(a[keep]);yt=tensor(yy);mt=tensor(m)
  for _ in range(120):
   net.train();err=loss(net(xt),yt,mt);opt.zero_grad();err.backward();opt.step()
  net.eval()
  with torch.no_grad():out[:,targets]=net(tensor(b)).cpu().numpy()*sd+mu
  saved.append({'state_dict':net.cpu().state_dict(),'targets':targets,'target_mean':mu,'target_sd':sd})
 torch.save({'models':saved,'input_scaler':scaler,'seed':seed},path)
 return out