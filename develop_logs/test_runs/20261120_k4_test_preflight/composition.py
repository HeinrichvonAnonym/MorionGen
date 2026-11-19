"""Test composition from registered metadata only; no test fitting or tuning."""
import collections, hashlib, json
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
registry=ROOT/'config/gold_k_20261115/trials.jsonl'
rows=[json.loads(s) for s in registry.read_text().splitlines()]
hold=[r for r in rows if r['partition']=='final_holdout']
included=[r for r in hold if r['status']=='included']
andy=[r for r in included if r['dataset']=='andy'];tv=[r for r in included if r['dataset']=='collaborate']
def task(r):
 s=r['source']['member']
 return 'Packaging' if '/Packaging/' in s else ('Board_Connection' if '/Board_Connection/' in s else 'Drilling')
def quant(v,q):
 v=sorted(v); x=(len(v)-1)*q;i=int(x)
 return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
durations=[(r['frames']-1)*r['dt'] for r in tv]
result={'manifest_sha256':hashlib.sha256(registry.read_bytes()).hexdigest(),'old_usage':json.loads((ROOT/'config/evalkit_k_validation_20261116/final_usage.json').read_text()),'counts':{'records':len(included),'andy':len(andy),'television':len(tv),'duplicate_aliases':sum(r['status']=='duplicate_alias' for r in hold)},'andy':{'subjects':dict(collections.Counter(r['subject'] for r in andy)),'setups':dict(collections.Counter(r['setup'] for r in andy)),'sequence_labels':dict(collections.Counter(r['id'].split('_Seq_')[1].split('_Trial')[0] for r in andy)),'reference_scope':'Current independent reference supports head/thorax marker frames. Existing whole-body fixture outside these regions is a prior, not whole-body ground truth.'},'television':{'subjects':dict(collections.Counter(r['subject'] for r in tv)),'tasks':dict(collections.Counter(task(r) for r in tv)),'task_subject':{t:dict(collections.Counter(r['subject'] for r in tv if task(r)==t)) for t in ['Board_Connection','Drilling','Packaging']},'sampling_dt':dict(collections.Counter(str(r['dt']) for r in tv)),'duration_s':{'total':sum(durations),'minimum':min(durations),'median':quant(durations,.5),'p95':quant(durations,.95),'maximum':max(durations)},'shorter_than_5s':sum(t<5 for t in durations),'shorter_than_1s':sum(t<1 for t in durations),'central_5s_total_nominal_s':sum(min(t,5) for t in durations),'eight_format_routes':len(tv)*8},'new_K4_test_fit_runs':0,'execution_state':'Awaiting candidate scope and window scope; no test motions used to choose K parameters.'}
out=ROOT/'develop_logs/test_runs/20261120_k4_test_preflight/composition.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='old_usage'},ensure_ascii=False,indent=2))
