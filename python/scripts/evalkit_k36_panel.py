"""Fixed eight-format panel plus original-domain anchor control."""
from evalkit_k36_domain import *
DEST=BASE/'evalkit_k36_panel_20261117'
PARAMETERS={'continue_joint_branch':True,'continuity_weight':.02,'hz':20,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.}
def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 rows=inventory();DEST.mkdir(exist_ok=True)
 f={'rows':rows,'parameters':PARAMETERS,'inputs':{r[k]:digest(r[k]) for r in rows for k in ['model','comparison','kinematics']},'implementation':implementation_key(),'runner':digest(__file__),'scope':'11 fixed development windows across eight formats; original-domain controls on three diagnosed failures; no holdout'}
 p=DEST/'freeze.json'
 if p.exists():assert read(p)==f
 else:atomic(p,f)
 tasks=[(r,'combined',{**PARAMETERS,'_output':str(DEST)}) for r in rows]
 tasks += [(r,'original',{**PARAMETERS,'_output':str(DEST)}) for r in rows if r['method']=='kimodo' and r['window'] in ['010','074','077']]
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,tasks))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
