"""Join the precommitted input-only choices to subsequently completed R audits."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).parent;monitor=HERE.parent/'20261118_k4_monitor'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=HERE/'candidate_budget_diagnostic_v1.json';prior=json.loads(source.read_text());rows=prior['rows'];evidence={str(source):sha(source)}
requests=monitor/'candidate_budget_requests_v1.json'
lookup={}
for folder in ['pool_candidates','composed_pool_candidates']:
 p=monitor/folder/'requested_review.json';v=json.loads(p.read_text());evidence[str(p)]=sha(p)
 assert v['requests_sha256']==sha(requests)
 for r in v['records']:lookup[r['run'],r['window'],r['method'],r['candidate']]=r
for r in rows:
 if r['independent_pass'] is None:
  a=lookup[r['run'],r['window'],r['method'],r['chosen']]
  if a['status']=='completed':
   r['independent_pass']=a['independent_pass'];r['independent_failures']=a['failures']
summary=dict(records=len(rows),unique_window_format_pairs=len({(r['window'],r['method']) for r in rows}),unique_original_windows=len({r['window'] for r in rows}),passed=sum(r['independent_pass'] is True for r in rows),failed=sum(r['independent_pass'] is False for r in rows),unknown=sum(r['independent_pass'] is None for r in rows))
output=HERE/'candidate_budget_acceptance_v1.json'
if output.exists():raise ValueError('Keep completed review immutable')
output.write_text(json.dumps(dict(summary=summary,rows=rows,evidence_sha256=evidence,promotion='none; retrospective selector audit only; same-version full656 still required'),indent=2)+'\n')
print(summary)
