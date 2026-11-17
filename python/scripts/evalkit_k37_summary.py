"""Compare K3.7 candidates, selected outputs and K3.6 without changing masks."""
from evalkit_k36_summary import summarize
from evalkit_k35_legs import *
DEST=ROOT/'config/evalkit_k37_20261117'
def main():
 roots={n:BASE/('evalkit_k37_'+n+'_20261117') for n in ['probe','chart','panel']}
 records={n:[summarize(p) for p in sorted(root.glob('*/[0-9]*/result.json'))] for n,root in roots.items()}
 differences=[];flags=[]
 for r in records['panel']:
  old=summarize(BASE/'evalkit_k36_panel_20261117/combined'/r['window']/'result.json')
  for side in ['left','right']:
   a=old['regions'][side]['selected'];b=r['regions'][side]['selected']
   delta={k:b[k]-a[k] for k in ['end_to_end','max_step','velocity_error_p95','coverage']}
   delta.update(foot_position_m=b['foot_position']['raw']-a['foot_position']['raw'],knee_p95=b['knee']['end_to_end']['p95_abs']-a['knee']['end_to_end']['p95_abs'])
   entry={'window':r['window'],'method':r['method'],'side':side,'difference_vs_k36_selected':delta};differences.append(entry)
   if delta['end_to_end']>1 or delta['max_step']>5 or delta['velocity_error_p95']>10 or delta['coverage']<-.01:flags.append(entry)
 frozen=read(roots['probe']/'freeze.json');assert all(digest(ROOT/p)==h for p,h in frozen['protected'].items())
 refs=read(ROOT/'config/evalkit_k34_20261117/experiment_freeze.json')['references'];assert all(digest(p)==h for p,h in refs.items())
 for root in roots.values():assert all(digest(p)==h for p,h in read(root/'freeze.json')['inputs'].items())
 result={'records':records,'differences':differences,'regression_flags':flags,'protected_unchanged':len(frozen['protected']),'references_unchanged':len(refs),'new_holdout_runs':0,'scope':'17 full-window ablations/regressions, 11 unique development windows, eight formats; candidate and fallback distinct; no full 82x8 regression'}
 atomic(DEST/'acceptance.json',result);atomic(DEST/'freeze.json',{k:read(v/'freeze.json') for k,v in roots.items()})
 print('flags',flags,'protected',result['protected_unchanged'],'references',len(refs))
 for r in records['panel']:
  print(r['window'],r['method'],{s:{'accepted':r['guards'][s]['accepted'],'knee':v['candidate']['knee']['end_to_end']['p95_abs'],'foot':v['candidate']['foot_position']['raw'],'coverage':v['candidate']['coverage']} for s,v in r['regions'].items()})
 print('position costs',[(r['window'],r['side'],round(r['difference_vs_k36_selected']['foot_position_m']*100,4)) for r in differences])
if __name__=='__main__':main()
