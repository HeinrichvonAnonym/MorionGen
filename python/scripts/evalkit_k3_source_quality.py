#!/usr/bin/env python3
"""Explain excluded BVH world coordinates without altering the frozen fit selection."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from evalkit_k3_audit import ROOT,REG,ROOTS,archive,parse_bvh


def inspect(row):
    src=row['source'];b=parse_bvh(archive(str(Path(ROOTS['collaborate'])/src['archive'])).read(src['member']))
    b.frames=b.frames[::max(1,int(np.ceil(len(b.frames)/240)))];local=b.local_positions(True,'absolute_channels')
    # Remove the root BEFORE FK, preserving source rotations and all child translations.
    relative=np.zeros_like(local);global_r=np.zeros((*local.shape[:2],3,3))
    for i,joint in enumerate(b.joints):
        channels=[(k,c) for k,c in enumerate(joint.channels) if c.lower().endswith('rotation')]
        order=''.join(c[0] for _,c in channels)
        r=Rotation.from_euler(order,b.frames[:,[joint.channel_start+k for k,c in channels]],degrees=True).as_matrix() if channels else np.broadcast_to(np.eye(3),(len(local),3,3))
        global_r[:,i]=r if joint.parent<0 else global_r[:,joint.parent]@r
        if joint.parent>=0:relative[:,i]=relative[:,joint.parent]+np.einsum('tij,tj->ti',global_r[:,joint.parent],local[:,i])
    p,_=b.forward_kinematics(True,translation_mode='absolute_channels');world64=relative+local[:,:1]
    return {'id':row['id'],'sampled_frames':len(local),'source_root_abs_max_m':float(np.max(np.abs(local[:,0]))*.01),
            'source_root_range_m':(np.ptp(local[:,0],axis=0)*.01).tolist(),
            'float32_world_fk_error_max_m':float(np.max(np.abs(p-world64))*.01),
            'root_relative_extent_m':(np.ptp(relative.reshape(-1,3),axis=0)*.01).tolist(),
            'local_pose_finite':bool(np.isfinite(relative).all() and np.isfinite(global_r).all()),
            'world_geometry_status':'excluded_extreme_source_root_translation',
            'local_pose_status':'recoverable_by_root_removal_before_fk',
            'scope':'diagnostic only; no repaired world trajectory, floor, or source recording added to frozen selection'}


def main():
    folder=ROOT/'config/evalkit_k3_20261116';bad={r['id'] for r in json.loads((folder/'light_summary.json').read_text())['failures']}
    rows=[json.loads(line) for line in (REG/'trials.jsonl').read_text().splitlines()]
    result={'selection_change':False,'cause':'extreme root translations already in source BVH; float32 world FK further loses limb-scale precision',
            'records':[inspect(r) for r in rows if r['id'] in bad]}
    (folder/'excluded_source_quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print(len(result['records']),'source exclusions diagnosed')

if __name__=='__main__':main()
