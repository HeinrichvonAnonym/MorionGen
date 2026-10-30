"""Compact score envelopes without hiding extrema or fabricating frame precision."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
snapshot=ROOT/'develop_logs/notes/motiongen_survey_revision/src/data.json'
rows=[]
for row in json.loads(snapshot.read_text())['queries']['supplement_posture']['rows']:
    source=Path(row['arrays'])
    if not source.is_absolute():source=ROOT/source
    arrays=np.load(source)
    fps=float(arrays['fps'])
    series={}
    for side in ['left_RULA','right_RULA','left_REBA','right_REBA']:
        bound=arrays[side+'_bounds'];level=arrays[side+'_action_level_bounds']
        bins=np.array_split(np.arange(len(bound)),min(120,len(bound)))
        series[side]={'intervals':[[int(b[0])/fps,(int(b[-1])+1)/fps,int(bound[b,0].min()),int(bound[b,1].max())] for b in bins],
                      'definite_high_action_seconds':float(np.sum(level[:,0]>=3)/fps),
                      'possible_high_action_seconds':float(np.sum(level[:,1]>=3)/fps),
                      'frames':len(bound)}
    rows.append({'motion_path':row['motion_path'],'method':row['method'],'source':str(source),'source_sha256':checksum(source),
                 'series':series})
json_write(BASE/'posture_score_curves.json',dict(rows=rows,
    scope='At most 120 time bins per motion: lower bound minimum and upper bound maximum within each bin. Not interpolated frame values. Exposure sums use all original frames, not binned previews. Bounds represent missing inputs, not confidence intervals.'))
print(json.dumps({'motions':len(rows)}))
