"""Frozen K3.9 complete development regression; reference scoring after fitting."""
from evalkit_k39_prepare import *
from motiogen.evalkit.k.config import implementation_key
CONFIG='config/evalkit_k39_20261117/runtime_config.json'


def score(row,work,selection):
 from motiogen.evalkit.k.trustworthy import directions
 from motiogen.evalkit.k.selection import selection_guard
 from motiogen.evalkit.k.temporal import temporal
 from motiogen.evalkit.k.scoring import stats,contact
 data=np.load(row['common']);R,D,t,L=data['R'],data['D'],data['times'],data['reference_lengths']
 fits={};supports={};full={}
 for name in ['baseline','candidate','selected']:
  z=np.load(work/name/'fit.npz');F=common(interp(z['times'],z['fitted'],t),list(z['names']));fits[name]=F;full[name]=z
  observed=common(interp(z['times'],z['source'],t),list(z['names']));np.testing.assert_allclose(observed,D,atol=1e-8)
  quality=apply_metric_gates(support(D,F,NAMES,5.,.05),metric_fidelity(D,F,NAMES,5.));supports[name]={k:v['angle_valid'] for k,v in quality.items()}
 scores,arrays=compare(R,D,{n:D for n in fits},fits,t,supports,float(np.percentile(R[:,[5,9],1],5)),reference_lengths=L)
 for name,F in fits.items():
  scores[name]['spatial_directions']={key:{stage:stats(np.degrees(np.arccos(np.clip((directions(a)[key]*directions(b)[key]).sum(-1),-1,1)))) for stage,a,b in [('degradation',D,R),('fit_deviation',F,D),('end_to_end',F,R)]} for key in directions(R)}
  scores[name]['direction_bias']={key:{stage:np.mean(directions(a)[key]-directions(b)[key],axis=0).tolist() for stage,a,b in [('degradation',D,R),('fit_deviation',F,D),('end_to_end',F,R)]} for key in directions(R)}
  scores[name]['direction_bias_definition']='mean unit-vector difference in world xyz; angular separation is unsigned, its mean is not a signed anatomical bias'
  scores[name]['temporal']={'input':temporal(D,F,t),'reference':temporal(R,F,t)}
  # Every finite scoring frame contributes to all three named coverage rates.
  scores[name]['position_coverage']={joint:{'scorable':float(np.mean(np.isfinite(F[:,j]).all(1)&np.isfinite(R[:,j]).all(1))),
    'input_fidelity_at_5cm':float(np.mean(np.linalg.norm(F[:,j]-D[:,j],axis=1)<=.05)),
    'end_to_end_at_5cm':float(np.mean(np.linalg.norm(F[:,j]-R[:,j],axis=1)<=.05))} for j,joint in enumerate(NAMES)}
  pre=common(interp(full[name]['times'],full[name]['pre_root_fitted'],t),list(full[name]['names']))
  scores[name]['root_effect']={'pre_root_pelvis_error_m':stats(np.linalg.norm(pre[:,0]-R[:,0],axis=1),False),
    'post_root_pelvis_error_m':stats(np.linalg.norm(F[:,0]-R[:,0],axis=1),False),
    'speed_change_m_s':stats(np.linalg.norm(np.gradient(F[:,0]-pre[:,0],t,axis=0),axis=1),False),
    'pre_root_contact_proxy':contact(pre,t,float(np.percentile(R[:,[5,9],1],5))),
    'post_root_contact_proxy':contact(F,t,float(np.percentile(R[:,[5,9],1],5))),
    'plane_scope':'reference-derived scoring plane only; never used by runtime fitting'}
  arrays['pre_root:'+name]=pre
 accepted,gate=selection_guard(R,fits['baseline'],fits['selected'],t)
 # Lack of further aggregate improvement is not itself a regression of fallback.
 regression=gate['failures']
 np.savez_compressed(work/'comparison.npz',**arrays)
 return {'scores':scores,'reference_regressions':regression,'arrays':str(work/'comparison.npz')}


def one(task):
 row,destination=task
 from motiogen.evalkit.k.selection import fit
 import opensim as osim
 osim.Logger.setLevelString('error');work=Path(destination)/row['window_key']/row['method'];work.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(work).free>80*1024**3,'80 GiB reserve'
 identity={k:row[k] for k in ['id','method','group_id','subset','window_key','source_window_s','reference','source','common']}
 receipt=work/'result.json'
 if receipt.exists():return str(receipt)
 start=__import__('time').monotonic()
 try:
  selection=fit(row['source'],work,CONFIG)
  result={'row':identity,'status':'computed','selection':selection,**score(row,work,selection)}
 except Exception as error:
  import traceback
  (work/'error.log').write_text(traceback.format_exc());result={'row':identity,'status':'failed','reason':str(error)}
 result['seconds']=__import__('time').monotonic()-start;atomic(receipt,result)
 print(row['window_key'],row['method'],result['status'],round(result['seconds'],1),result.get('reason',''),flush=True)
 return str(receipt)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--selected-only',action='store_true');a=parser.parse_args();a.out.mkdir(exist_ok=True)
 rows=read(OUT/'exports.json');rows=[r for r in rows if r['selected'] or not a.selected_only]
 rows.sort(key=lambda r:(0 if (r['window_key'],r['method'])==('074','kimodo') else 1 if r['selected'] else 2,r['window_key'],r['method']))
 frozen={'implementation':implementation_key(),'config':read(ROOT/CONFIG),'runner':digest(__file__),'rows':[(r['method'],r['window_key']) for r in rows],
  'inputs':{r['common']:digest(r['common']) for r in rows},'scope':'fixed development only; full original windows; no held-out data'}
 path=a.out/'freeze.json'
 if path.exists():assert read(path)==frozen,'frozen identity changed; use new version directory'
 else:atomic(path,frozen)
 with ProcessPoolExecutor(max_workers=min(a.workers,3)) as pool:results=list(pool.map(one,[(r,a.out) for r in rows]))
 atomic(a.out/'index.json',results)
 assert frozen['implementation']==implementation_key(),'implementation changed during experiment'
 history=read(OUT/'source_freeze.json');changed=[p for p,h in history['protected'].items() if digest(ROOT/p)!=h]
 changed_references=[p for p,h in history['references'].items() if digest(p)!=h]
 atomic(a.out/'integrity.json',{'changed_historical_files':changed,'changed_references':changed_references,'free_bytes':shutil.disk_usage(a.out).free,'results':len(results),'expected':len(rows)})
 assert not changed and not changed_references
if __name__=='__main__':main()
