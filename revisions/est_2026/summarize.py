from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from scipy.stats import spearmanr
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,matthews_corrcoef,average_precision_score
R=Path(__file__).resolve().parent;NC=R.parents[1];O=R/'summary';O.mkdir(exist_ok=True)
def metrics(d):
 y=d.observed.to_numpy();p=d.predicted.to_numpy();err=np.abs(y-p);g=d.assign(err=err).groupby('compound_id').err.mean()
 return {'n':len(d),'chemicals':len(g),'studies':d.study_id.nunique() if 'study_id' in d else None,'MAE':float(err.mean()),'chemical_MAE':float(g.mean()),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)) if len(y)>1 else None,'rho':float(spearmanr(y,p).statistic) if len(set(y))>1 and len(set(p))>1 else None}
def aggregate(d,keys):
 rows=[]
 for vals,g in d.groupby(keys,dropna=False):
  if not isinstance(vals,tuple):vals=(vals,)
  rows.append(dict(zip(keys,vals),**metrics(g)))
 return pd.DataFrame(rows)
def paired(d,groupcols,comparators):
 out=[];rng=np.random.default_rng(20260924)
 for vals,g in d.groupby(groupcols):
  if not isinstance(vals,tuple):vals=(vals,)
  for model in comparators:
   a=g[g.model.eq('full_multitask')];b=g[g.model.eq(model)];m=a.merge(b,on=['record_id','compound_id'],suffixes=('_f','_b'),validate='one_to_one')
   if m.empty:continue
   m['delta']=abs(m.predicted_f-m.observed_f)-abs(m.predicted_b-m.observed_b);c=m.groupby('compound_id').delta.mean().to_numpy();draw=np.mean(rng.choice(c,(5000,len(c))),axis=1)
   out.append(dict(zip(groupcols,vals),comparator=model,n=len(m),chemicals=len(c),delta=float(c.mean()),low=float(np.quantile(draw,.025)),high=float(np.quantile(draw,.975))))
 return pd.DataFrame(out)
manifest=[]
for folder,jobfile in [('results_final','final_jobs.json'),('results_similarity','similarity_jobs.json')]:
 jobs=json.loads((R/'data'/jobfile).read_text());missing=[];allrows=[]
 for job in jobs:
  name=job.get('name',job.get('task'));f=R/folder/(name+'.csv');a=R/folder/(name+'_audit.json')
  if not f.exists() or not a.exists():missing.append(name);continue
  m=json.loads(a.read_text());d=pd.read_csv(f);assert hashlib.sha256(f.read_bytes()).hexdigest()==m['result_sha256'];assert m['final_data_sha256']==hashlib.sha256((R/'data/final_observations.csv').read_bytes()).hexdigest();assert set(m['train_ids']).isdisjoint(m['test_ids'])
  if m['design']=='chemical':assert m['chemical_intersection']==0
  if m['design']=='study':assert m['study_intersection']==0
  assert np.isfinite(d.predicted.to_numpy(dtype=float)).all();assert not d.duplicated(['record_id','model']).any()
  for k in m['upstream_keys']:
   z=json.loads((R/'checkpoints'/f'transport_{k}.json').read_text());assert set(m['excluded_compounds'])<=set(z['excluded_compounds'])
  manifest.append({'task':name,'rows':len(d),'skipped':m['skipped'],'chemical_intersection':m['chemical_intersection'],'study_intersection':m['study_intersection']});allrows.append(d)
 assert not missing,(folder,missing)
 z=pd.concat(allrows,ignore_index=True);z.to_csv(O/(folder+'_predictions.csv'),index=False)
 keys=['cohort','design','model','endpoint','seed'];aggregate(z,keys).to_csv(O/(folder+'_by_seed.csv'),index=False)
 ens=z.groupby(['cohort','design','model','endpoint','record_id','compound_id','study_id'],as_index=False).agg(observed=('observed','first'),predicted=('predicted','mean'),seeds=('seed','nunique'))
 ens.to_csv(O/(folder+'_ensemble_predictions.csv'),index=False);aggregate(ens,['cohort','design','model','endpoint']).to_csv(O/(folder+'_metrics.csv'),index=False)
 paired(ens,['cohort','design','endpoint'],['structure_context','shuffled_auxiliary_labels','full_single_task','extra_trees_structure_context']).to_csv(O/(folder+'_paired.csv'),index=False)
(O/'validation_manifest.json').write_text(json.dumps(manifest,indent=2))
e=pd.concat([pd.read_csv(p) for p in (R/'external_run/results').glob('source_transfer_*.csv')]+[pd.read_csv(R/'external_run/results/strict_crossfitted_predictions.csv')],ignore_index=True);assert not e.duplicated(['mode','model','seed','record_id']).any();assert e.predicted.notna().all()
e['endpoint']=e.endpoint_type;e['observed']=e.log10_value_L_kg;e['panel_label']=np.where(e.panel.eq('external_BCF'),np.where(e.tissue.eq('liver'),'Yao_liver_BCF','Hayman_whole_body_BCF'),e.panel)
ens=e.groupby(['mode','panel_label','model','endpoint','record_id','compound_id','study_id'],as_index=False).agg(observed=('observed','first'),predicted=('predicted','mean'),seeds=('seed','nunique'));ens.to_csv(O/'external_ensemble_predictions.csv',index=False);aggregate(ens,['mode','panel_label','model','endpoint']).to_csv(O/'external_metrics.csv',index=False);paired(ens,['mode','panel_label','endpoint'],['structure_context','extra_trees']).to_csv(O/'external_paired.csv',index=False)
# Endpoint-compatible external tool outputs retained with their original applicability flags.
op=pd.read_csv(R/'data/opera_source_matched_records.csv');op=op[op.tissue.eq('whole body')].copy();op['observed']=op.log10_value_L_kg;op['endpoint']='BCF';op['mode']='archived_tool_execution';op['panel_label']='Hayman_whole_body_BCF';op['model']='OPERA_2.9.5';assert op.record_id.is_unique
common=set(op.record_id)&set(ens[ens.panel_label.eq('Hayman_whole_body_BCF')].record_id);assert len(common)==27
op.to_csv(O/'opera_wholebody_archived.csv',index=False);aggregate(pd.concat([ens[ens.record_id.isin(common)],op]),['mode','model']).to_csv(O/'opera_comparison.csv',index=False)
cl=[]
for (mode,model),g in ens[ens.panel_label.eq('Hayman_whole_body_BCF')].groupby(['mode','model']):
 for threshold in [np.log10(2000),np.log10(5000)]:
  y=(g.observed>threshold).astype(int);p=(g.predicted>threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();sens=tp/(tp+fn) if tp+fn else np.nan;spec=tn/(tn+fp) if tn+fp else np.nan
  cl.append({'mode':mode,'model':model,'threshold':threshold,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),'accuracy':float((y==p).mean()),'majority_accuracy':float(max(y.mean(),1-y.mean())),'sensitivity':sens,'specificity':spec,'balanced_accuracy':(sens+spec)/2,'MCC':float(matthews_corrcoef(y,p)),'average_precision':float(average_precision_score(y,g.predicted)) if y.sum() else None,'positive_compounds':g.loc[y.eq(1),'compound_id'].nunique()})
pd.DataFrame(cl).to_csv(O/'threshold_diagnostics.csv',index=False)
cal=pd.concat([pd.read_csv(R/f'decision_run/results/calibration_{s}.csv') for s in [42,43,44]]);cal.to_csv(O/'calibration.csv',index=False)
cp=pd.concat([pd.read_csv(R/f'decision_run/results/calibration_predictions_{s}.csv') for s in [42,43,44]]);cp['ae']=abs(cp.observed-cp.predicted);cp.to_csv(O/'calibration_predictions.csv',index=False)
sel=[];bins=[]
for (seed,ep),g in cp.groupby(['seed','endpoint']):
 c=g.groupby('compound_id').agg(uncertainty=('uncertainty','mean'),MAE=('ae','mean')).sort_values('uncertainty')
 for n in range(1,len(c)+1):
  h=g[g.compound_id.isin(c.index[:n])];sel.append({'seed':seed,'endpoint':ep,'retained_fraction':n/len(c),'retained_compounds':n,'record_fraction':len(h)/len(g),'chemical_MAE':c.MAE.iloc[:n].mean(),'record_MAE':h.ae.mean()})
 for rank,ids in enumerate(np.array_split(c.index.to_numpy(),min(3,len(c)))):
  h=g[g.compound_id.isin(ids)];bins.append({'seed':seed,'endpoint':ep,'uncertainty_group':rank+1,'chemicals':len(ids),'mean_uncertainty':h.uncertainty.mean(),'MAE':h.ae.mean(),'RMSE':np.sqrt(np.mean(h.ae**2))})
pd.DataFrame(sel).to_csv(O/'selective_prediction.csv',index=False);pd.DataFrame(bins).to_csv(O/'uncertainty_error_bins.csv',index=False)
print('CORE_SUMMARY_COMPLETE',len(manifest),flush=True)
