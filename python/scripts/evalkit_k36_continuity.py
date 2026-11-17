"""Same expanded model and input: joint-branch continuation and sampling ablation."""
from evalkit_k36_domain import *
DEST=BASE/'evalkit_k36_continuity_20261117'
VARIANTS={'continue':{'continue_joint_branch':True},'regularized':{'continue_joint_branch':True,'continuity_weight':.02},'dense':{'continue_joint_branch':True,'continuity_weight':.02,'hz':100}}
def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 rows=[r for r in inventory() if r['method']=='kimodo' and r['window'] in ['010','074','077']];DEST.mkdir(exist_ok=True)
 freeze={'rows':rows,'variants':VARIANTS,'profile':'combined','inputs':{r[k]:digest(r[k]) for r in rows for k in ['model','comparison','kinematics']},'implementation':implementation_key(),'runner':digest(__file__),'domain_runner':digest(ROOT/'python/scripts/evalkit_k36_domain.py'),'scope':'same model/input full-window temporal diagnostic; no reference-driven selection'}
 p=DEST/'freeze.json'
 if p.exists():assert read(p)==freeze
 else:atomic(p,freeze)
 tasks=[(r,'combined',{**settings,'_output':str(DEST/name)}) for name,settings in VARIANTS.items() for r in rows]
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,tasks))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
