"""Dated PubMed candidate retrieval using the recovered original query strings."""
from pathlib import Path
import argparse,datetime,json,time,urllib.parse,urllib.request
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
queries=json.loads((Path(__file__).parent/'queries.json').read_text())
manifest={'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'database':'PubMed','purpose':'supplementary reproducible candidate retrieval; not original historical search','queries':[]};all_ids=set()
for name,query in queries.items():
 params={'db':'pubmed','term':query,'retmode':'json','retmax':10000,'tool':'PFAS_revision_reproducibility'}
 url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urllib.parse.urlencode(params)
 for attempt in range(5):
  try:
   with urllib.request.urlopen(url,timeout=60) as r:raw=r.read()
   obj=json.loads(raw)['esearchresult'];break
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
 (out/(name+'.json')).write_bytes(raw)
 count=int(obj['count']);ids=obj['idlist']
 if count>10000 or len(ids)!=count:raise RuntimeError('Incomplete retrieval: narrow/date-partition this query before claiming completeness')
 all_ids.update(ids);manifest['queries'].append({'name':name,'query':query,'count':count,'retrieved':len(ids),'url':url})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));time.sleep(.4)
manifest['unique_pmids']=len(all_ids)
(out/'pmids.txt').write_text('\n'.join(sorted(all_ids,key=int))+'\n')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
