"""Recompute headline metrics from supplied predictions; no model fitting or GPU."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--out',type=Path);a=p.parse_args()
 root=a.root.resolve();out=a.out or root/'reproduced';out.mkdir(parents=True,exist_ok=True)
 groups=[('results_final',['cohort','design','model','endpoint']),('results_similarity',['cohort','design','model','endpoint']),('external',['mode','panel_label','model','endpoint'])]
 rows=[];checks=[]
 for stem,keys in groups:
  data=pd.read_csv(root/'summary'/f'{stem}_ensemble_predictions.csv',float_precision='round_trip');expected=pd.read_csv(root/'summary'/f'{stem}_metrics.csv',float_precision='round_trip')
  assert data[['observed','predicted']].notna().all().all()
  for vals,g in data.groupby(keys):
   y=g.observed.to_numpy();pred=g.predicted.to_numpy();err=abs(y-pred);c=g.assign(error=err).groupby('compound_id').error.mean()
   den=((y-y.mean())**2).sum();r2=1-((y-pred)**2).sum()/den if den else np.nan
   # Round-trip CSV parsing preserves exact ties in the archived float64 predictions.
   rho=spearmanr(y,pred).statistic if len(np.unique(y))>1 and len(np.unique(pred))>1 else np.nan
   r=dict(zip(keys,vals),n=len(g),chemicals=len(c),MAE=err.mean(),chemical_MAE=c.mean(),RMSE=np.sqrt(((y-pred)**2).mean()),R2=r2,rho=rho)
   mask=np.ones(len(expected),bool)
   for k,v in zip(keys,vals):mask &= expected[k].eq(v).to_numpy()
   ref=expected[mask];assert len(ref)==1,(stem,vals)
   ref=ref.iloc[0]
   for metric in ['n','chemicals','MAE','chemical_MAE','RMSE','R2','rho']:
    assert np.isclose(r[metric],ref[metric],rtol=0,atol=1e-10,equal_nan=True),(stem,vals,metric,r[metric],ref[metric])
   rows.append(dict(dataset=stem,**r));checks.append({'dataset':stem,'group':list(vals),'records':len(g),'pass':True})
 pd.DataFrame(rows).to_csv(out/'recomputed_metrics.csv',index=False)
 (out/'verification.json').write_text(json.dumps({'all_passed':True,'groups':checks,'tolerance':1e-10},indent=2))
 print(f'PASS: {len(checks)} metric groups independently reproduced from prediction CSVs.')

if __name__=='__main__':main()
