from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'summary/results_final_ensemble_predictions.csv');d=d[(d.cohort=='primary216')&(d.design=='chemical')];rng=np.random.default_rng(20260924);out=[]
for ep in ['BAF','BCF']:
 for comp in ['structure_context','shuffled_auxiliary_labels','extra_trees_structure_context']:
  a=d[(d.endpoint==ep)&d.model.eq('full_multitask')];b=d[(d.endpoint==ep)&d.model.eq(comp)];m=a.merge(b,on=['record_id','compound_id','study_id'],suffixes=('_a','_b'));m['delta']=abs(m.predicted_a-m.observed_a)-abs(m.predicted_b-m.observed_b);nc=m.compound_id.nunique();ns=m.study_id.nunique()
  if ns<2:out.append({'endpoint':ep,'comparator':comp,'status':'one source, joint interval not estimable','compounds':nc,'studies':ns});continue
  ci=pd.factorize(m.compound_id)[0];si=pd.factorize(m.study_id)[0];v=m.delta.to_numpy();vals=[]
  for _ in range(5000):
   wc=rng.multinomial(nc,np.ones(nc)/nc);ws=rng.multinomial(ns,np.ones(ns)/ns)[si];num=np.bincount(ci,weights=v*ws,minlength=nc);den=np.bincount(ci,weights=ws,minlength=nc);ok=(den>0)&(wc>0)
   if ok.any():vals.append(np.average(num[ok]/den[ok],weights=wc[ok]))
  out.append({'endpoint':ep,'comparator':comp,'status':'product multinomial sensitivity','compounds':nc,'studies':ns,'delta':m.groupby('compound_id').delta.mean().mean(),'low':np.quantile(vals,.025),'high':np.quantile(vals,.975),'valid_replicates':len(vals)})
pd.DataFrame(out).to_csv(R/'summary/paired_joint_sensitivity.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
