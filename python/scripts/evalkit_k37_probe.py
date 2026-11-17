"""Development-only 074 leg chart/domain ablation on immutable K3.6 inputs."""
from evalkit_k36_domain import *
from copy import deepcopy
DEST=BASE/'evalkit_k37_probe_20261117'
PARAMETERS={'continue_joint_branch':True,'continuity_weight':.02,'hz':20,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.}
PROBES={}
for name in ['hip_chart','ankle_chart','both_charts']:
 spec=deepcopy(PROFILES['combined'])
 if name in ['hip_chart','both_charts']:spec['ranges'].update(hip_flexion=(-360,360),hip_adduction=(-180,180),hip_rotation=(-360,360))
 if name in ['ankle_chart','both_charts']:spec['ranges'].update(ankle_angle=(-180,180),subtalar_angle=(-180,180))
 PROBES[name]=spec
PROFILES.update(PROBES)
def main():
 DEST.mkdir(exist_ok=True);row=next(r for r in inventory() if r['window']=='074')
 frozen={'row':row,'profiles':PROBES,'parameters':PARAMETERS,'inputs':{row[k]:digest(row[k]) for k in ['model','comparison','kinematics']},'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},'implementation':implementation_key(),'runner':digest(__file__),'scope':'kinematic coordinate-domain diagnosis; no anatomical or force claim; no holdout'}
 atomic(DEST/'freeze.json',frozen)
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_domain,[(row,p,{**PARAMETERS,'_output':str(DEST)}) for p in PROBES]))
 atomic(DEST/'index.json',results)
if __name__=='__main__':main()
