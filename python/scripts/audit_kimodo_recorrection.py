"""Replay only official correction on saved raw motions, without diffusion/GPU."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.config import load_config
from motiogen.runtime import checksum,json_write
DATA=Path(load_config()['data_root'])
sys.path.insert(0,str(DATA/'baselines/kimodo'))
from kimodo.skeleton import build_skeleton
from kimodo.constraints import load_constraints_lst
from kimodo.postprocess import post_process_motion

p=argparse.ArgumentParser();p.add_argument('--limit',type=int)
p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
p.add_argument('--receipt',type=Path)
args=p.parse_args()
torch.set_num_threads(1)
skeleton=build_skeleton(30).to(args.device)
skeleton.somaskel77.to(args.device)
source=ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
rows=json.loads(source.read_text())['rows']
if args.limit:rows=rows[:args.limit]
base=DATA/'benchmarks/kimodo-stratified-20261008'
checks=[]
for row in rows:
    raw=base/'generated_raw'/row['case'];post=base/'generated_postprocess'/row['case']
    assert checksum(raw/'motion.npz')==row['outputs']['raw']['sha256']
    assert checksum(post/'motion.npz')==row['outputs']['postprocess']['sha256']
    with np.load(raw/'motion.npz',allow_pickle=False) as saved:
        rotations=torch.from_numpy(saved['local_rot_mats'].copy())
        roots=torch.from_numpy(saved['root_positions'].copy())
        contacts=torch.from_numpy(saved['foot_contacts'].copy())
    if rotations.ndim==4:rotations=rotations[None];roots=roots[None];contacts=contacts[None]
    rotations=rotations.to(args.device);roots=roots.to(args.device);contacts=contacts.to(args.device)
    compact=skeleton.from_SOMASkeleton77(rotations)
    # Exact inverse of output_to_SOMASkeleton77's duplicated toe-end channels.
    contacts=contacts[...,[0,1,3,4]]
    constraints=load_constraints_lst(str(raw/'constraints.json'),skeleton,device=torch.device(args.device))
    corrected=post_process_motion(compact,roots,contacts,skeleton,[constraints],root_margin=.04)
    corrected['foot_contacts']=contacts
    expanded=skeleton.output_to_SOMASkeleton77(corrected)
    fields=[]
    with np.load(post/'motion.npz',allow_pickle=False) as expected:
        for key in ['root_positions','local_rot_mats','global_rot_mats','posed_joints','foot_contacts']:
            actual=expanded[key].cpu().numpy()
            target=expected[key]
            if actual.ndim==target.ndim+1:actual=actual[0]
            assert actual.shape==target.shape,(key,actual.shape,target.shape)
            delta=float(np.max(actual!=target)) if actual.dtype==np.bool_ else float(np.max(np.abs(actual-target)))
            tolerance=0. if key=='foot_contacts' else (2e-5 if key in ['root_positions','posed_joints'] else 5e-5)
            fields.append({'field':key,'max_abs_difference':delta,'tolerance':tolerance,'passed':delta<=tolerance})
    checks.append({'case':row['case'],'fields':fields,'passed':all(f['passed'] for f in fields)})
    if len(checks)%32==0:print(json.dumps({'checked':len(checks),'passed':sum(c['passed'] for c in checks)}),flush=True)
receipt={'passed':all(c['passed'] for c in checks),'checked_pairs':len(checks),'expected_pairs':416,'device':args.device,
         'checks':checks,'source_audit_sha256':checksum(source),
         'scope':'Official correction replay from persisted raw rotations/root/contact, exact native SOMA77-to-30 inverse and 30-to-77 output conversion. No diffusion rerun or retraining. Device is recorded explicitly. Unchanged foot contact channels require exact numeric equality; position tolerance 2e-5 m and rotation-matrix tolerance 5e-5 are fixed across CPU/GPU checks.'}
out=args.receipt or ROOT/'develop_logs/test_runs/20261024_kimodo_recorrection'/('audit.json' if args.limit is None else f'limit{args.limit}.json')
json_write(out,receipt)
print(json.dumps({'passed':receipt['passed'],'checked':len(checks),'failures':[c for c in checks if not c['passed']],'output':str(out)}))
if args.device=='cuda' and not receipt['passed']:raise SystemExit(1)
