"""Isolate copied-model knee/hip/ankle domains on fixed development windows."""
from evalkit_k35_legs import *
from motiogen.evalkit.k.model_domain import prepare,PROFILES
OUT=BASE/'evalkit_k36_domain_20261117'

def run_domain(task):
 row,profile,*options=task;overrides=dict(options[0]) if options else {};destination=Path(overrides.pop('_output',OUT));work=destination/profile/row['window'];work.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(OUT).free>80*1024**3
 import opensim as osim
 osim.Logger.setLevelString('error');z=np.load(row['kinematics']);t=z['times'];labels=list(z['labels']);names=list(z['names']);initial=z['coordinates'];source=z['source'];model=work/'model.osim'
 initial,domain=prepare(row['model'],model,labels,initial,profile)
 settings={**SETTINGS,'foot_weight':1.,'max_nfev':100,**overrides};selected,raw,receipt=refine_legs(model,t,labels,initial,source,names,settings)
 old=np.load(row['comparison']);D=old['D'];R=old['R'];F=old[row['fit_key']];assert np.allclose(t,old['times'])
 fits={};masks={}
 for key,q in [('baseline',initial),('candidate',raw),('selected',selected)]:
  p,ns=positions(model,t,labels,q);fits[key]=common(p,ns);quality=apply_metric_gates(support(D,fits[key],NAMES),metric_fidelity(D,fits[key],NAMES,10));masks[key]={k:v['angle_valid'] for k,v in quality.items()}
 np.testing.assert_allclose(fits['baseline'],F,atol=1e-6)
 scores,arrays=compare(R,D,{k:D for k in fits},fits,t,masks,float(np.nanpercentile(D[:,[5,9],1],5)),reference_lengths=old['reference_lengths'])
 np.savez_compressed(work/'comparison.npz',**arrays);np.savez_compressed(work/'coordinates.npz',times=t,labels=labels,baseline=initial,candidate=raw,selected=selected)
 report={'row':row,'profile':profile,'domain':domain,'receipt':receipt,'scores':scores,'work':str(work)};atomic(work/'result.json',report)
 print(profile,row['window'],{s:r['accepted'] for s,r in receipt['legs'].items()},flush=True)
 return {'profile':profile,'window':row['window'],'path':str(work/'result.json')}

def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 OUT.mkdir(exist_ok=True);rows=[r for r in inventory() if r['method']=='kimodo' and r['window'] in ['010','074','077']]
 frozen={'rows':rows,'profiles':PROFILES,'inputs':{r[k]:digest(r[k]) for r in rows for k in ['model','comparison','kinematics']},'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},'implementation':implementation_key(),'runner':digest(__file__),'scope':'three fixed development windows; range/unlock diagnosis; no production model change; no holdout'}
 p=OUT/'freeze.json'
 if p.exists():assert read(p)==frozen
 else:atomic(p,frozen)
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,[(r,p) for r in rows for p in PROFILES]))
 atomic(OUT/'index.json',results)
if __name__=='__main__':main()
