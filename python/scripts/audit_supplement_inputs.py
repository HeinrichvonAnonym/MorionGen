"""Inventory persisted angle inputs, not a claim that rotations identify anatomy."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write,checksum
from motiogen.supplement_inputs import accepted_inputs

rows=[];summary=defaultdict(Counter)
for row in accepted_inputs():
    path=Path(row['motion_path']);record={k:row[k] for k in ['method','task','seed','clock','condition','run','motion_path','input_format']}
    try:
        with np.load(path,allow_pickle=False) as motion:
            shapes={k:list(motion[k].shape) for k in motion.files if k!='metadata_json'}
            names=motion['joint_names'].tolist() if 'joint_names' in motion else []
        raw=Path(row['run'])/'raw/native_prediction.npz'
        if row['method']=='primal':raw=Path(row['run'])/'factorial'/f"{row['variant']}.npz"
        raw_shapes={}
        if raw.exists():
            with np.load(raw,allow_pickle=False) as data:raw_shapes={k:list(data[k].shape) for k in data.files}
        record.update(status='read',sha256=checksum(path),fields=shapes,joint_names=names,raw_prediction_fields=raw_shapes,
                      rotations_persisted=any('rot' in k or k=='pose' for k in shapes|raw_shapes),
                      external_load='not_registered',work_cycle='not_registered',coupling='not_registered',
                      anatomy_scope='Rotation presence is not anatomical calibration; finger/hand channels require model-source audit.')
        summary[row['method']]['read']+=1
        summary[row['method']]['rotations_persisted']+=record['rotations_persisted']
    except Exception as exc:
        record.update(status='read_failed',error=f'{type(exc).__name__}: {exc}');summary[row['method']]['read_failed']+=1
    rows.append(record)
    if len(rows)%100==0:print(json.dumps({'checked':len(rows),'summary':summary}),flush=True)
out=ROOT/'develop_logs/test_runs/20261028_supplement/input_audit.json'
json_write(out,{'rows':rows,'summary':summary,'scope':'2400 primary cells plus two conditions of 416 historical Kimodo pairs; repeated clocks are not independent samples.'})
print(json.dumps({'checked':len(rows),'summary':summary}),flush=True)
