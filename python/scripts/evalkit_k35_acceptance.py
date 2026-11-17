"""Reapply final all-bone temporal guards to saved candidates, never refit to R."""
from evalkit_k35_legs import *
from motiogen.evalkit.k.legs import accept_window

def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args();OUT=a.out
 report=[]
 for item in read(OUT/'panel.json'):
  record=read(item['path']);z=np.load(Path(record['work'])/'comparison.npz');t=z['times'];D=z['D'];R=z['R'];old=z['F:baseline'];raw=z['F:candidate'];selected=old.copy();receipts={}
  for side in ['left','right']:
   ids=[NAMES.index(side+'_'+n) for n in ['hip','knee','ankle','foot']]
   yes,r=accept_window(old[:,ids],raw[:,ids],D[:,ids],t,SETTINGS);receipts[side]={'accepted':yes,**r}
   if yes:selected[:,ids]=raw[:,ids]
  fits={'baseline':old,'candidate':raw,'selected':selected};masks={}
  for k,v in fits.items():
   q=apply_metric_gates(support(D,v,NAMES),metric_fidelity(D,v,NAMES,10));masks[k]={n:part['angle_valid'] for n,part in q.items()}
  scores,arrays=compare(R,D,{k:D for k in fits},fits,t,masks,float(np.nanpercentile(D[:,[5,9],1],5)),reference_lengths=z['reference_lengths'])
  dest=OUT/'accepted'/item['window']/item['method'];dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'comparison.npz',**arrays)
  summary={}
  ar=angles(R)
  for reg in ['left_leg','right_leg']:
   keys=[k for k in ar if region(k)==reg];summary[reg]={}
   for k,v in fits.items():
    af=angles(v);summary[reg][k]={'e2e_p95_mean':float(np.mean([scores[k]['raw']['angles'][m]['end_to_end']['p95_abs'] for m in keys])),
     'max_step':max(float(abs(np.diff(af[m])).max()) for m in keys),'velocity_error_p95':float(np.mean([np.percentile(abs(np.gradient(af[m],t)-np.gradient(ar[m],t)),95) for m in keys])), 'coverage':float(masks[k][reg].mean())}
  result={**item,'guards':receipts,'summary':summary,'scores':scores,'output':str(dest)};atomic(dest/'result.json',result);report.append(result)
  print(item['method'],item['window'],{s:r['accepted'] for s,r in receipts.items()},flush=True)
 frozen=read(OUT/'freeze.json');assert all(digest(ROOT/p)==h for p,h in frozen['protected'].items());assert all(digest(p)==h for p,h in frozen['inputs'].items())
 atomic(OUT/'acceptance.json',{'records':report,'guards':SETTINGS,'experiment_commit':'e580516','scope':'11 fixed development windows, original R scores kept separate from input-only selection; full original time windows','protected_unchanged':len(frozen['protected']),'input_files_unchanged':len(frozen['inputs']),'new_holdout_runs':0})
if __name__=='__main__':main()
