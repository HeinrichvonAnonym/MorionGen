"""Same-input regression of the minimal continuous hip-flexion chart."""
from evalkit_k36_domain import *
DEST=BASE/'evalkit_k37_panel_20261117'
PARAMETERS={'continue_joint_branch':True,'continuity_weight':.02,'hz':20,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.}
def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 DEST.mkdir(exist_ok=True);rows=inventory()
 frozen={'rows':rows,'profile':PROFILES['hip_flexion_chart'],'parameters':PARAMETERS,'inputs':{r[k]:digest(r[k]) for r in rows for k in ['model','comparison','kinematics']},'implementation':implementation_key(),'runner':digest(__file__),'scope':'11 fixed development windows, eight formats; full original R scoring; no holdout'}
 p=DEST/'freeze.json'
 if p.exists():assert read(p)==frozen
 else:atomic(p,frozen)
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,[(r,'hip_flexion_chart',{**PARAMETERS,'_output':str(DEST)}) for r in rows]))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
