"""Anonymous multi-view posture package; the key never enters the rater HTML."""
import hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
from motiogen.supplement_inputs import accepted_inputs
from motiogen.posture_geometry import read_positions,observable_angles
out=ROOT/'develop_logs/test_runs/20261028_supplement/blind_rating';out.mkdir(exist_ok=True)
rng=random.Random(20261028);groups=defaultdict(list)
for row in accepted_inputs():groups[row['method']].append(row)
base_edges=[('pelvis','spine1'),('spine1','spine2'),('spine2','neck'),('neck','head')]
for side in ['left','right']:
    base_edges += [('pelvis',side+'_hip'),(side+'_hip',side+'_knee'),(side+'_knee',side+'_ankle'),(side+'_ankle',side+'_foot'),('neck',side+'_shoulder'),(side+'_shoulder',side+'_elbow'),(side+'_elbow',side+'_wrist')]
pack=[];key=[]
for method,rows in sorted(groups.items()):
    # Exclude identical persisted motions (not merely duplicated filenames).
    unique={}
    for row in rows:
        path=Path(row['motion_path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in unique:unique[digest]=row
    candidates=list(unique.values());rng.shuffle(candidates)
    for ordinal,row in enumerate(candidates[:48]):
        xyz,names,fps=read_positions(row);angles=observable_angles(xyz,names)
        if ordinal<32:frame=rng.randrange(len(xyz));stratum='uniform_frame'
        else:
            boundary=[60.,100.][ordinal%2]
            frame=int(np.argmin(abs(angles[['left_elbow_flexion_deg','right_elbow_flexion_deg'][ordinal%2]]-boundary)));stratum='elbow_boundary_enriched'
        id=hashlib.sha256(f'{method}/{ordinal}/{rng.random()}'.encode()).hexdigest()[:12]
        idx={n:i for i,n in enumerate(names)};edges=[[idx[a],idx[b]] for a,b in base_edges if a in idx and b in idx]
        for side in ['Left','Right']:
            wrist=side.lower()+'_wrist'
            for stem in ['HandThumb','HandIndex','HandMiddle','HandRing','HandPinky']:
                chain=[wrist]+[side+stem+s for s in ['1','2','3','4','End']]
                valid=[n for n in chain if n in idx]
                edges.extend([[idx[a],idx[b]] for a,b in zip(valid,valid[1:])])
            if side+'HandEnd' in idx:edges.append([idx[wrist],idx[side+'HandEnd']])
        neighbours=np.unique(np.clip(np.array([frame-round(fps*.5),frame,frame+round(fps*.5)]),0,len(xyz)-1))
        poses=xyz[neighbours].astype(float);poses-=poses[:,idx['pelvis']:idx['pelvis']+1,:]
        pack.append({'id':id,'poses':poses.round(5).tolist(),'edges':edges,'selected':int(np.flatnonzero(neighbours==frame)[0]),
            'left_indices':[i for i,n in enumerate(names) if n.lower().startswith('left')],
            'right_indices':[i for i,n in enumerate(names) if n.lower().startswith('right')],
            'wrist_indices':{s:idx[s+'_wrist'] for s in ['left','right'] if s+'_wrist' in idx},
            'offset_seconds':((neighbours-frame)/fps).tolist(),'task_conditions':'外部负载、支撑、握持和职业重复频度没有登记，请标为无法判读。此处只提供骨架观测。橙色为左侧，紫色为右侧，蓝色为中轴。'})
        key.append({'blind_id':id,'method':method,'run':row['run'],'task':row.get('task'),'seed':row.get('seed'),
            'condition':row.get('condition'),'motion_path':row['motion_path'],'frame':frame,'stratum':stratum,
            'cluster':str(row.get('cluster',row.get('task'))),'sampling_note':'Frames from one task/source remain correlated; repeated seeds/conditions are not new independent source actions.'})
rng.shuffle(pack)
json_write(out.parent/'blind_rating_private_key.json',key)
json_write(out/'sampling_design.json',{'poses':len(pack),'per_method_target':48,'random_frames_per_method':32,'boundary_enriched_per_method':16,
    'precision_target':{'confidence':.95,'halfwidth':.05,'independent_worst_case_normal_approx_n':385},
    'not_a_sample_size_guarantee':'385 is only the unclustered Bernoulli planning bound. This 384-posture pilot is clustered and boundary-enriched. Estimate source-cluster dependence and stratum agreement, then expand/reweight for the precision target; no completed human rating is claimed.',
    'human_rating_status':'pending_no_raters','private_key':'Stored outside rater package; do not send it to raters.'})
template=(ROOT/'utils/motiogen/blind_rating_template.html').read_text()
(out/'index.html').write_text(template.replace('__POSE_DATA__',json.dumps(pack,separators=(',',':'))))
print(json.dumps({'poses':len(pack),'html':str(out/'index.html')}))
