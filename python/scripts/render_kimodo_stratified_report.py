#!/usr/bin/env python3
"""Render matched Kimodo GT/raw/correction snapshots on one fixed 3D scale."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.motion import SOMA77_NAMES,canonicalize

CHAINS=(
    ('pelvis','spine1','spine2','spine3','neck','head'),
    ('pelvis','left_hip','left_knee','left_ankle','left_foot'),
    ('pelvis','right_hip','right_knee','right_ankle','right_foot'),
    ('neck','left_shoulder','left_elbow','left_wrist'),
    ('neck','right_shoulder','right_elbow','right_wrist'),
)
DEFAULT_GROUPS=(
    'content/text2motion/overview',
    'content/constraints_notext/end-effectors/hands_posrot',
    'content/constraints_withtext/fullbody/random',
    'repetition/constraints_withtext/root/path_2dposrot',
)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def load_joints(path):
    with np.load(path,allow_pickle=False) as data:
        joints=np.asarray(data['posed_joints'])
    if joints.ndim==4: joints=joints[0]
    joints,_=canonicalize(joints,'Y-up',units='m')
    return joints


def frame_for(case,nframes):
    path=Path(case)/'constraints.json'
    if path.is_file():
        for item in json.loads(path.read_text()):
            frames=item.get('frame_indices',[])
            if frames: return min(int(frames[len(frames)//2]),nframes-1),'constraint frame'
    return nframes//2,'sequence midpoint'


def choose_cases(raw,post,groups):
    chosen=[]
    for group in groups:
        candidates=sorted((Path(raw)/group).glob('*/motion.npz'))
        paired=sorted((Path(post)/group).glob('*/motion.npz')) if post else []
        if paired: candidates=[Path(raw)/path.parent.relative_to(post)/'motion.npz' for path in paired]
        if candidates: chosen.append(str(candidates[len(candidates)//2].parent.relative_to(raw)))
    return chosen


def render(raw,post,case,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base=Path(raw)/case; paths=[('GT',base/'gt_motion.npz'),('Raw',base/'motion.npz')]
    corrected=Path(post)/case/'motion.npz' if post else None
    if corrected and corrected.is_file(): paths.append(('Correction',corrected))
    motions=[(label,path,load_joints(path)) for label,path in paths]
    frame,reason=frame_for(base,min(len(x[2]) for x in motions))
    origin=motions[0][2][frame,0].copy(); origin[2]=0
    index={name:i for i,name in enumerate(SOMA77_NAMES)}
    fig=plt.figure(figsize=(5.2*len(motions),5.2),constrained_layout=True)
    colors=('#3157d5','#4d6d42','#c85218','#8661c1','#d44a42')
    for panel,(label,path,joints) in enumerate(motions,1):
        ax=fig.add_subplot(1,len(motions),panel,projection='3d'); points=joints[frame]-origin
        ax.scatter(*points.T,s=7,color='#77849a',alpha=.55)
        for chain,color in zip(CHAINS,colors):
            ids=[index[name] for name in chain]
            ax.plot(*points[ids].T,color=color,linewidth=2.5,marker='o',markersize=2.5)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),zlim=(0,2.2),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect((2.8,2.8,2.2)); ax.view_init(elev=18,azim=-58); ax.tick_params(labelsize=7)
        ax.set_title(f'{label} · frame {frame}\nX/Y ±1.4 m · Z 0–2.2 m',fontsize=10)
    fig.suptitle(f'{case} · {reason} · common GT-pelvis origin',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=145,facecolor='white'); plt.close(fig)
    return {'case':case,'frame':frame,'frame_reason':reason,'snapshot':str(output),
            'motions':[{'label':label,'path':str(path),'sha256':sha256(path)} for label,path,_ in motions]}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--raw',required=True); parser.add_argument('--postprocess')
    parser.add_argument('--output',required=True); parser.add_argument('--case',action='append',default=[])
    args=parser.parse_args(argv); raw=Path(args.raw); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    cases=args.case or choose_cases(raw,args.postprocess,DEFAULT_GROUPS); rows=[]
    for case in cases:
        slug=case.replace('/','__')+'.png'; rows.append(render(raw,args.postprocess,case,output/slug))
    manifest={'schema_version':1,'fixed_view':{'xlim_m':[-1.4,1.4],'ylim_m':[-1.4,1.4],'zlim_m':[0,2.2]},
              'origin':'matched GT pelvis at rendered frame','cases':rows}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'success','cases':len(rows),'output':str(output)},ensure_ascii=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
