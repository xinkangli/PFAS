from pathlib import Path
import sys,json,re,math,hashlib,sqlite3,shutil
import pandas as pd,numpy as np,openpyxl
R=Path('/DATA/lxk/lxkzero/claude/qd');B=R/'data/nc/old/water_research_core_20260925';O=R/'data/nc/old/water_research_v17_source_closure'
for f in ['audit','data','code','documents','sources']:(O/f).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'disi'))
from collectors.itrc import parse_xlsx,ITRC_FILES
raw=R/'data/04_bioaccumulation/ITRC_Table_5-1_BCF-BAF_aquatic.xlsx';rawhash=hashlib.sha256(raw.read_bytes()).hexdigest()
records=parse_xlsx(raw,next(c for c in ITRC_FILES if c['label']=='aquatic'));lookup={}
for r in records:
 m=re.search(r'uid=([^;]+)',r.get('notes',''))
 if m:lookup.setdefault((m[1].strip(),r['endpoint_type']),[]).append(r)
d=pd.read_csv(B/'data/final_observations.csv');assert d.record_id.is_unique and len(d)==1078
wb=openpyxl.load_workbook(raw,data_only=True);ws=wb['BCF-BAF Database'];headers={str(c.value).strip():c.column for c in ws[7] if c.value is not None}
c=sqlite3.connect('file:'+str(R/'data/04_bioaccumulation/pfas_master.sqlite')+'?mode=ro',uri=True);db=pd.read_sql_query('select * from pfas_bcf_master',c);c.close();dbidx=db.set_index('id')
joined=pd.read_excel(R/'data/model_data/Model2_Bioaccumulation.xlsx',sheet_name='X_Y_joined');joined_idx=joined.set_index('id')
rows=[]
for _,v in d.iterrows():
 uid=re.search(r'uid=([^;]+)',v.database_locator)[1].strip();rn=int(v.v9_row);ep=v.endpoint_type;matches=lookup.get((uid,ep),[])
 cells={k:ws.cell(rn,headers[k]).value for k in [ep,ep+' Min',ep+' Max'] if k in headers};vs={k:pd.to_numeric(x,errors='coerce') for k,x in cells.items()};direct=vs.get(ep,np.nan);lo=vs.get(ep+' Min',np.nan);hi=vs.get(ep+' Max',np.nan)
 if pd.notna(direct):derived=float(direct);rule='reported_database_value'
 elif pd.notna(lo) and pd.notna(hi):derived=(float(lo)+float(hi))/2;rule='arithmetic_midpoint_of_reported_range'
 elif pd.notna(lo):derived=float(lo);rule='minimum_only_used_as_point'
 elif pd.notna(hi):derived=float(hi);rule='maximum_only_used_as_point'
 else:derived=np.nan;rule='unresolved'
 parser_ok=len(matches)==1 and math.isclose(matches[0]['value'],float(v.value),rel_tol=1e-9,abs_tol=1e-7)
 cell_ok=math.isclose(derived,float(v.value),rel_tol=1e-9,abs_tol=1e-7)
 dbrow=dbidx.loc[v.id];jrow=joined_idx.loc[v.id]
 db_ok=math.isclose(dbrow.value,float(v.value),rel_tol=1e-9,abs_tol=1e-7) and dbrow.endpoint_type==ep and dbrow.source=='ITRC'
 joined_ok=math.isclose(jrow.value,float(v.value),rel_tol=1e-9,abs_tol=1e-7) and jrow.endpoint_type==ep
 unverified=v.v15_status in ['unverified_primary_table','table_located_aggregation_not_resolved','table_located_value_not_resolved']
 if v.v15_primary:tier='primary_included';reason='Source-qualified endpoint and value under frozen eligibility criteria'
 elif unverified:tier='database_traced_primary_not_verified';reason={'unverified_primary_table':'Original-paper numerical cell not verified','table_located_aggregation_not_resolved':'Original table located but aggregation cannot be reconstructed','table_located_value_not_resolved':'Original table located but numerical mapping unresolved'}[v.v15_status]
 else:tier='primary_located_not_eligible';reason=str(v.v8_action)+'; '+str(v.v8_note)
 if v.v15_status=='source_summary_verified_censoring_sensitivity':reason='Censoring substitutions contribute to published means; per-cell reconstruction unavailable'
 if v.v15_status=='verified_plasma_unit_mismatch':reason='Plasma ratios are dimensionless and incompatible with mass-based L/kg regression'
 rows.append(dict(record_id=v.record_id,id=v.id,endpoint=ep,compound_id=v.compound_id,citation=v.citation,doi=v.doi_normalized,raw_file=str(raw),raw_sha256=rawhash,sheet=ws.title,row=rn,cells=';'.join(ws.cell(rn,headers[k]).coordinate for k in cells),raw_cells=json.dumps(cells,ensure_ascii=False,default=str),conversion_rule=rule,reconstructed_value=derived,historical_value=v.value,current_log10=v.log10_value_L_kg,original_parser_match=parser_ok,raw_cell_match=cell_ok,sqlite_match=db_ok,model_workbook_match=joined_ok,tier=tier,primary_included=bool(v.v15_primary),original_table_status=v.v15_status,original_table_locator=v.v15_locator,exclusion_reason=reason))
a=pd.DataFrame(rows)
for col in ['original_parser_match','raw_cell_match','sqlite_match','model_workbook_match']:assert a[col].all(),a.loc[~a[col],'record_id'].tolist()
assert a.tier.value_counts().to_dict()=={'database_traced_primary_not_verified':696,'primary_included':216,'primary_located_not_eligible':166}
a.to_csv(O/'audit/all_1078_disposition.csv',index=False)
a[a.tier.eq('database_traced_primary_not_verified')].to_csv(O/'audit/excluded_696_database_traced.csv',index=False)
a[a.tier.eq('primary_located_not_eligible')].to_csv(O/'audit/excluded_166_endpoint_or_value.csv',index=False)
a.groupby(['tier','citation','doi','original_table_status'],dropna=False).agg(records=('record_id','size'),compounds=('compound_id','nunique')).reset_index().to_csv(O/'audit/study_disposition.csv',index=False)
d[d.v15_primary].to_csv(O/'data/primary_216_only.csv',index=False)
d[~d.v15_primary].to_csv(O/'data/archive_862_not_for_modeling.csv',index=False)
# Current admission boundary: no excluded biological label enters any primary job.
primary=set(d.loc[d.v15_primary,'record_id']);excluded=set(d.loc[~d.v15_primary,'record_id']);tasks=[]
for file in ['final_jobs.json','similarity_jobs.json']:
 for job in json.loads((B/'data'/file).read_text()):
  tr=set(job['train_ids']);te=set(job['test_ids']);assert tr|te<=primary;assert not tr&te;assert not (tr|te)&excluded
  tasks.append({'task':job['task'],'train':len(tr),'test':len(te),'excluded_overlap':0})
pred=pd.read_csv(B/'summary/results_final_predictions.csv');sim=pd.read_csv(B/'summary/results_similarity_predictions.csv');assert set(pred.record_id)|set(sim.record_id)<=primary
pd.DataFrame(tasks).to_csv(O/'audit/current_partition_admission_check.csv',index=False)
# Verify the actual 69 BCF target values against the saved original-table audit.
bcf=d[d.v15_primary & d.endpoint_type.eq('BCF')];source=pd.read_csv(R/'data/nc/revision/v15_validation_20260924/audit/new_primary_cell_verification.csv');check=bcf.merge(source[['record_id','source_log10','locator','primary_eligible']],on='record_id',validate='one_to_one')
assert len(check)==69 and check.primary_eligible_y.all();assert np.allclose(check.log10_value_L_kg,check.source_log10,atol=1e-12,rtol=0)
assert (bcf.v9_conversion_rule=='reported_database_value').all()
check[['record_id','endpoint_type','tissue','exposure_group','log10_value_L_kg','source_log10','locator']].to_csv(O/'audit/BCF_69_original_log_targets.csv',index=False)
flow=[]
j=joined.copy();flow.append({'stage':'Original model workbook','records':len(j)})
j=j[j.endpoint_type.isin(['BAF','BCF','BCFD'])];flow.append({'stage':'Recognized endpoint','records':len(j)})
j=j[pd.to_numeric(j.log_value,errors='coerce').notna()];flow.append({'stage':'Numeric original log value','records':len(j)})
j=j[j.SMILES.notna()];flow.append({'stage':'Nonempty SMILES','records':len(j)})
from rdkit import Chem
j=j[j.SMILES.map(lambda v: Chem.MolFromSmiles(str(v)) is not None)];flow.append({'stage':'Parsable structure','records':len(j)})
factor=j.value_units.map({'L/kg':1.,'ml/g':1.,'L/g':1000.,'L/mg':1000000.});val=pd.to_numeric(j.value,errors='coerce')*factor
j=j[np.isfinite(val)&(val>0)];flow.append({'stage':'Confirmed positive L/kg conversion','records':len(j)})
j=j[j.basis.fillna('').str.lower().isin(['wet weight','ww'])];flow.append({'stage':'Wet-weight basis','records':len(j)})
j=j[~j.notes.fillna('').str.contains(r'\bNDs?\b|\bMDL\b|\bLOD\b|\bLOQ\b|non.detect|below detection',case=False,regex=True)];flow.append({'stage':'No unresolved censoring note in initial screen','records':len(j)})
assert set(j.id)==set(d.id), (len(j),len(set(j.id)-set(d.id)),len(set(d.id)-set(j.id)))
pd.DataFrame(flow).to_csv(O/'audit/historical_training_filter_counts.csv',index=False)
summary={'historical_rows':1078,'database_lineage_replayed':1078,'unresolved_original_table':696,'unresolved_studies':int(a[a.tier.eq('database_traced_primary_not_verified')].doi.nunique()),'primary_table_located_not_eligible':166,'primary_included':216,'primary_compounds':int(d[d.v15_primary].compound_id.nunique()),'primary_studies':int(d[d.v15_primary].doi_normalized.nunique()),'excluded_total':862,'original_parser_rows':len(records),'historical_range_midpoints':int((a.conversion_rule=='arithmetic_midpoint_of_reported_range').sum()),'range_midpoints_in_primary':int((a.primary_included & a.conversion_rule.eq('arithmetic_midpoint_of_reported_range')).sum()),'primary_BCF_source_exact_log_matches':69,'partition_checks':len(tasks),'excluded_records_in_current_partitions':0,'raw_sha256':rawhash,'scientific_results_changed':False,'historical_filter_flow':flow}
(O/'audit/closure_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for f in ['disi/collectors/itrc.py','disi/schema.py','disi/pfas_targets.py','data/build_model_data.py','data/train_model2.py','data/nc/revision/src/audit_data.py','data/nc/revision/v9/reconstruct_lineage.py','data/nc/revision/v15_validation_20260924/audit_sources.py']:
 p=R/f;dest=O/'code'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(raw,O/'sources'/raw.name)
(O/'audit/source_code_hashes.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'code').rglob('*.py')},indent=2))
print(json.dumps(summary,indent=2))
