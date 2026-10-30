"""Identifiable posture and explicit score bounds for the shared generation pilots."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.motion import load_motion
from motiogen.posture_geometry import observable_angles,episodes
from motiogen.posture_scoring import score_angle_series
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
OUT=BASE/'generation_posture';OUT.mkdir(exist_ok=True)
rows=[]
for sample in json.loads((BASE/'generation_evaluation_with_references.json').read_text())['rows']:
    source=Path(sample['output'])/'motion.npz'
    motion=load_motion(source);xyz=motion['joints'];fps=float(motion['fps'])
    angles=observable_angles(xyz,motion['joint_names'].tolist())
    arrays,scores=score_angle_series(angles,fps)
    target=OUT/f"{sample['method']}_{sample['source_id']}_{sample['seed']}.npz"
    np.savez_compressed(target,**arrays)
    rows.append(dict(method=sample['method'].lower(),task='generation:'+sample['source_id'],seed=sample['seed'],
        condition={'native_text_generation':True},motion_path=str(source),source_sha256=checksum(source),
        status='position_components_computed',frames=len(xyz),fps=fps,ergonomic=scores,arrays=str(target),
        angles={k:{'min':float(np.nanmin(v)),'max':float(np.nanmax(v))} for k,v in angles.items()},
        exposure={side+'_upper_arm_above90':episodes(angles[side+'_upper_arm_elevation_deg']>90,fps) for side in ['left','right']},
        channel='HumanML3D native generated joint positions; no observed hand orientation',human_validation='pending_no_raters'))
json_write(OUT/'analysis.json',dict(rows=rows,
    scope='32 native generation pilots, same source text and seed identifiers as the shared evaluator. Elbow/knee components and position-observable motion are retained; task conditions and unobserved hand orientation remain unknown.'))
print(json.dumps({'motions':len(rows)}))
