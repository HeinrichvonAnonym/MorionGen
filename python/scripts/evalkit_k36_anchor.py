"""Input-conditioned bidirectional continuation; same fixed model and windows."""
from evalkit_k36_domain import *
DEST=BASE/'evalkit_k36_anchor_20261117'
VARIANTS={'anchor20':{'continue_joint_branch':True,'continuity_weight':.02,'hz':20,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.},'anchor100':{'continue_joint_branch':True,'continuity_weight':.02,'hz':100,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.}}
def main():
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 rows=[r for r in inventory() if r['method']=='kimodo' and r['window'] in ['010','074','077']];DEST.mkdir(exist_ok=True)
 f={'rows':rows,'variants':VARIANTS,'profile':'combined','inputs':{r[k]:digest(r[k]) for r in rows for k in ['model','comparison','kinematics']},'implementation':implementation_key(),'runner':digest(__file__),'scope':'offline input-only anchor; original full-window scoring; no holdout'}
 p=DEST/'freeze.json'
 if p.exists():assert read(p)==f
 else:atomic(p,f)
 tasks=[(r,'combined',{**settings,'_output':str(DEST/name)}) for name,settings in VARIANTS.items() for r in rows]
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,tasks))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
