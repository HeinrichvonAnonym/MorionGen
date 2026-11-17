"""Fresh public-API old/new-input paired regression, separate from fixed-pose ablation."""
from evalkit_k38_root import *
from motiogen.evalkit.k.domain_fitting import fit_kinematic_domain
OUT=BASE/'evalkit_k38_runtime_20261117'
CONFIG_PATH='config/evalkit_k38_20261117/kinematic_config.json'
def run(kind):
 import opensim as osim
 osim.Logger.setLevelString('error')
 original=read(BASE/'evalkit_k31_full_20261116/074/kimodo/result.json')
 source=original['source'] if kind=='old_input' else str(DEST/'074/kimodo/corrected_input')
 work=OUT/kind
 result=fit_kinematic_domain(source,work,config=CONFIG_PATH,method='kimodo')
 z=np.load(work/'kinematic_fit.npz');previous=np.load(BASE/'evalkit_k37_panel_20261117/hip_flexion_chart/074/comparison.npz');t=previous['times'];R=previous['R'];lengths=previous['reference_lengths'];names=list(z['names'])
 D=common(interp(z['times'],z['source'],t),names);record={}
 for variant,field in [('before_root','pre_root_fitted'),('aligned','fitted')]:
  F=common(interp(z['times'],z[field],t),names);record[variant],arrays=score(R,D,F,t,lengths);np.savez_compressed(work/(variant+'.npz'),**arrays)
 repeat=fit_kinematic_domain(source,work,config=CONFIG_PATH,method='kimodo');np.testing.assert_array_equal(result['coords'],repeat['coords'])
 assert not result['force_eligible'] and not (work/'model.osim').exists()
 report={'kind':kind,'source':source,'scores':record,'receipt':result['receipt'],'cache_replay_equal':True,'output':str(work),'scope':'fresh K3.4 nominal + K3.7 leg fit + K3.8 root alignment; compare old/new inputs under same implementation; separate from frozen K3.1 nominal'}
 atomic(work/'result.json',report)
 print(kind,{k:round(s['raw']['positions']['left_foot']['end_to_end']['world']['p95_abs']*100,4) for k,s in record.items()},flush=True)
 return str(work/'result.json')
def main():
 OUT.mkdir(exist_ok=True)
 frozen={'implementation':implementation_key(),'config':read(ROOT/CONFIG_PATH),'runner':digest(__file__),'scope':'074 development only, two complete public API runs; original clock scoring; no holdout'}
 p=OUT/'freeze.json'
 if p.exists():assert read(p)==frozen
 else:atomic(p,frozen)
 with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['old_input','corrected_input']))
 atomic(OUT/'index.json',results)
if __name__=='__main__':main()
