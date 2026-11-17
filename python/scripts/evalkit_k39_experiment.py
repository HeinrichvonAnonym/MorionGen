"""Paired fixed-geometry / solver experiment on common native inputs."""
from evalkit_k39_prepare import *
from motiogen.evalkit.k.trustworthy import fit_variant,directions,guard
from motiogen.evalkit.k.scoring import stats
CONFIG='config/evalkit_k39_20261117/experiment_config.json'

def one(task):
 row,destination,variants=task
 import opensim as osim
 osim.Logger.setLevelString('error')
 work=Path(destination)/row['window_key']/row['method'];work.mkdir(parents=True,exist_ok=True)
 result=work/'result.json';started=__import__('time').monotonic();records={}
 try:
  old=np.load(row['common']);t=old['times'];R=old['R'];D=old['D'];lengths=old['reference_lengths']
  for name in variants:
   shape,solver=name=='g1s0' or name=='g1s1',name=='g0s1' or name=='g1s1'
   rec=fit_variant(row['source'],work/name,CONFIG,shape,solver,nominal_work=work/'nominal')
   z=np.load(work/name/'fit.npz');F=common(interp(z['times'],z['fitted'],t),list(z['names']));pre=common(interp(z['times'],z['pre_root_fitted'],t),list(z['names']))
   observed=common(interp(z['times'],z['source'],t),list(z['names']));np.testing.assert_allclose(observed,D,atol=1e-8)
   quality=apply_metric_gates(support(D,F,NAMES,5.,.05),metric_fidelity(D,F,NAMES,5.));masks={k:v['angle_valid'] for k,v in quality.items()}
   scores,arrays=compare(R,D,{'candidate':D,'pre_root':D},{'candidate':F,'pre_root':pre},t,{'candidate':masks,'pre_root':masks},float(np.percentile(R[:,[5,9],1],5)),reference_lengths=lengths)
   for variant,p in [('candidate',F),('pre_root',pre)]:
    scores[variant]['spatial_directions']={key:{stage:stats(np.degrees(np.arccos(np.clip((directions(a)[key]*directions(b)[key]).sum(-1),-1,1)))) for stage,a,b in [('degradation',D,R),('fit_deviation',p,D),('end_to_end',p,R)]} for key in directions(R)}
   np.savez_compressed(work/name/'comparison.npz',**arrays)
   records[name]={'scores':scores,'receipt':str(work/name/'receipt.json'),'arrays':str(work/name/'comparison.npz'),'position_coverage':{k:float(v['position_valid'].mean()) for k,v in quality.items()}}
  report={'row':row,'status':'computed','variants':records,'seconds':__import__('time').monotonic()-started,'work':str(work)}
 except Exception as error:
  import traceback
  (work/'error.log').write_text(traceback.format_exc());report={'row':row,'status':'failed','reason':str(error),'seconds':__import__('time').monotonic()-started,'work':str(work)}
 atomic(result,report);print(row['window_key'],row['method'],report['status'],round(report['seconds'],1),report.get('reason',''),flush=True);return str(result)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--pilot',action='store_true');parser.add_argument('--full',action='store_true');parser.add_argument('--variants',default='g0s0,g1s0,g0s1,g1s1');a=parser.parse_args();a.out.mkdir(exist_ok=True)
 rows=read(OUT/'exports.json');rows=[r for r in rows if r['selected'] or a.full]
 if a.pilot:rows=[r for r in rows if r['method']=='kimodo' and r['window_key']=='074']
 frozen={'implementation':implementation_key(),'config':read(ROOT/CONFIG),'rows':[(r['method'],r['window_key']) for r in rows],'variants':a.variants,'runner':digest(__file__),'inputs':{str(r['common']):digest(r['common']) for r in rows},'scope':'fixed development, same-input comparison, no holdout'}
 p=a.out/'freeze.json'
 if p.exists():assert read(p)==frozen,'Use a new output directory'
 else:atomic(p,frozen)
 with ProcessPoolExecutor(max_workers=min(3,len(rows))) as pool:results=list(pool.map(one,[(r,a.out,a.variants.split(',')) for r in rows]))
 atomic(a.out/'index.json',results)
if __name__=='__main__':main()
