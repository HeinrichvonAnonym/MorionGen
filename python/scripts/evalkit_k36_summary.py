"""Summarize domain and continuation evidence without changing score windows."""
from evalkit_k36_domain import *
from motiogen.evalkit.k.config import digest
ROOTS={'domain':BASE/'evalkit_k36_domain_20261117','continuity':BASE/'evalkit_k36_continuity_20261117','anchor':BASE/'evalkit_k36_anchor_20261117','panel':BASE/'evalkit_k36_panel_20261117'}
def summarize(path):
 r=read(path);z=np.load(Path(r['work'])/'comparison.npz');ar=angles(z['R']);t=z['times'];regions={}
 for side in ['left','right']:
  reg=side+'_leg';names=[n for n in ar if region(n)==reg];regions[side]={}
  for variant in ['baseline','candidate','selected']:
   scores=r['scores'][variant];af=angles(z['F:'+variant]);regions[side][variant]={
    stage:float(np.mean([scores['raw']['angles'][n][stage]['p95_abs'] for n in names])) for stage in ['degradation','fit_deviation','end_to_end']}
   regions[side][variant].update(coverage=scores['coverage'][reg]['share'],max_step=max(float(abs(np.diff(af[n])).max()) for n in names),velocity_error_p95=float(np.mean([np.percentile(abs(np.gradient(af[n],t)-np.gradient(ar[n],t)),95) for n in names])),
     knee={stage:scores['raw']['angles'][side+'_knee_flexion'][stage] for stage in ['degradation','fit_deviation','end_to_end']},foot_position={space:scores[space]['positions'][side+'_foot']['end_to_end']['world']['p95_abs'] for space in ['raw','normalized']},contact=scores['contact']['F:'+variant][side])
 return {'path':str(path),'method':r['row']['method'],'window':r['row']['window'],'source_version':r['row']['version'],'profile':r['profile'],'settings':r['receipt']['settings'],'guards':r['receipt']['legs'],'regions':regions,'domain':r['domain'],'frames':len(t)}
def main():
 allrecords={name:[summarize(p) for p in sorted(root.rglob('result.json'))] for name,root in ROOTS.items()}
 primary=read(ROOTS['domain']/'freeze.json');assert all(digest(ROOT/p)==h for p,h in primary['protected'].items())
 refs=read(ROOT/'config/evalkit_k34_20261117/experiment_freeze.json')['references'];assert all(digest(p)==h for p,h in refs.items())
 for root in ROOTS.values():
  frozen=read(root/'freeze.json');assert all(digest(p)==h for p,h in frozen['inputs'].items())
 result={'records':allrecords,'protected_unchanged':len(primary['protected']),'reference_files_unchanged':len(refs),'new_holdout_runs':0,'scope':'kinematic-only copied-model research; R scores complete original windows; input-only guards; no muscle-domain validation'}
 atomic(BASE/'evalkit_k36_summary_20261117.json',result)
 for name,rows in allrecords.items():print(name,len(rows))
 for r in allrecords['panel']:
  print(r['profile'],r['method'],r['window'],{s:{'accepted':r['guards'][s]['accepted'],'knee_before':round(v['baseline']['knee']['end_to_end']['p95_abs'],2),'knee_after':round(v['selected']['knee']['end_to_end']['p95_abs'],2),'region_before':round(v['baseline']['end_to_end'],2),'region_after':round(v['selected']['end_to_end'],2),'coverage':round(v['selected']['coverage'],3)} for s,v in r['regions'].items()})
if __name__=='__main__':main()
