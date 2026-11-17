"""Root-translation audit: immutable inputs, explicit R/D/F and 2x2 controls."""
from evalkit_k35_legs import *
from evalkit_k33_conversion import reference,interp
from motiogen.evalkit.k.root_alignment import align_fitted_root,hip_midpoint
from motiogen.evalkit.k.export import retarget,serialize,ALIASES
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.scoring import stats
DEST=BASE/'evalkit_k38_root_20261117'

def score(R,D,F,t,lengths):
 quality=apply_metric_gates(support(D,F,NAMES),metric_fidelity(D,F,NAMES,10))
 masks={k:v['angle_valid'] for k,v in quality.items()}
 scores,arrays=compare(R,D,{'output':D},{'output':F},t,{'output':masks},float(np.percentile(R[:,[5,9],1],5)),reference_lengths=lengths)
 scores=scores['output'];scores['position_coverage']={k:float(v['position_valid'].mean()) for k,v in quality.items()}
 scores['root_velocity']={label:stats(np.linalg.norm(np.gradient(a[:,0]-b[:,0],t,axis=0),axis=-1),False) for label,a,b in [('degradation',D,R),('fit_deviation',F,D),('end_to_end',F,R)]}
 return scores,arrays

def one(row):
 import opensim as osim
 osim.Logger.setLevelString('error')
 previous=BASE/'evalkit_k37_panel_20261117/hip_flexion_chart'/row['window'];work=DEST/row['window']/row['method'];work.mkdir(parents=True,exist_ok=True)
 z=np.load(previous/'comparison.npz');q=np.load(previous/'coordinates.npz');src=np.load(row['kinematics']);t=z['times'];labels=list(q['labels']);names=list(src['names']);D=z['D'];R=z['R'];old=z['F:selected'];model=previous/'model.osim'
 coords,fitted,valid,shift,receipt=align_fitted_root(model,t,labels,q['selected'],src['source'],names)
 F=common(fitted,names);np.testing.assert_allclose(F,old+shift[:,None],atol=1e-8)
 np.testing.assert_allclose(F[:,0],D[:,0],atol=1e-8)
 np.testing.assert_allclose(np.diff(F,axis=1),np.diff(old,axis=1),atol=1e-8)
 variants={'old':(D,old),'fit_anchor_only':(D,F)};extra={}
 if row['window']=='074':
  original=read(BASE/'evalkit_k31_full_20261116/074/kimodo/result.json');ref=reference(original)
  oldtarget,parents=retarget(ref,'kimodo',root_anchor='source_root')
  target,parents=retarget(ref,'kimodo',root_anchor='hip_midpoint');path=serialize(target,parents,work/'corrected_input');decoded=load(path,'kimodo')
  Dnew=common(interp(decoded.times-decoded.times[0],decoded.positions,t),decoded.names)
  Dcheck=common(interp(oldtarget.times-oldtarget.times[0],oldtarget.positions,t),oldtarget.names)
  np.testing.assert_allclose(Dcheck,D,atol=1e-10)
  delta=Dnew[:,0]-D[:,0];np.testing.assert_allclose(Dnew,D+delta[:,None],atol=1e-10)
  newsource=src['source']+delta[:,None]
  # Translation-equivariant controlled output: fixed internal coordinates;
  # model FK is recomputed rather than editing the evaluation arrays.
  combined,full,fullvalid,fullshift,fullreceipt=align_fitted_root(model,t,labels,q['selected'],newsource,names)
  combinedF=common(full,names)
  np.testing.assert_allclose(combinedF,F+delta[:,None],atol=1e-8)
  variants.update(export_anchor_only=(Dnew,old+delta[:,None]),both_anchors=(Dnew,combinedF))
  extra={'root_export':target.provenance['root_anchor'],'serialization_max_m':float(abs(decoded.positions-target.positions).max()),'paired_model_fk_max_m':float(abs(combinedF-(F+delta[:,None])).max()),'scope':'2x2 translation-only control on fixed K3.7 internal pose; distinct from full reoptimization'}
  np.savez_compressed(work/'both_coordinates.npz',times=t,labels=labels,coordinates=combined)
 results={}
 for key,(d,f) in variants.items():
  results[key],arrays=score(R,d,f,t,z['reference_lengths']);np.savez_compressed(work/(key+'.npz'),**arrays)
 np.savez_compressed(work/'root_coordinates.npz',times=t,labels=labels,old=q['selected'],aligned=coords,valid=valid,shift=shift)
 result={'row':row,'scores':results,'root_receipt':receipt,'extra':extra,'work':str(work)};atomic(work/'result.json',result)
 print(row['window'],{k:{'root':[round(v['raw']['positions']['pelvis_mid'][stage]['world']['p95_abs']*100,3) for stage in ['degradation','fit_deviation','end_to_end']], 'foot':round(v['raw']['positions']['left_foot']['end_to_end']['world']['p95_abs']*100,3)} for k,v in results.items()},flush=True)
 return str(work/'result.json')

def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 DEST.mkdir(exist_ok=True);assert shutil.disk_usage(DEST).free>80*1024**3;rows=inventory()
 inputs={r[k]:digest(r[k]) for r in rows for k in ['kinematics','comparison','model']}
 for row in rows:
  folder=BASE/'evalkit_k37_panel_20261117/hip_flexion_chart'/row['window']
  inputs.update({str(p):digest(p) for p in folder.iterdir() if p.is_file()})
 refs=read(ROOT/'config/evalkit_k34_20261117/experiment_freeze.json')['references']
 frozen={'rows':rows,'inputs':inputs,'references':refs,'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},'implementation':implementation_key(),'runner':digest(__file__),'scope':'11 fixed dev windows; 074 root-anchor 2x2, remaining windows input-only root audit; no holdout'}
 path=DEST/'freeze.json'
 if path.exists():assert read(path)==frozen,'Use a new output directory for changed implementation'
 else:atomic(path,frozen)
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,rows))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
