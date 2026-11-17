"""Original-reference plots for the saved 000 Kimodo diagnostic case."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frozen=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    d=load(frozen/'inputs/000/kimodo/candidate','kimodo')
    with np.load(frozen/'references/000.npz') as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
    paths={'Common':a.run/'diagnostics_v2/000/kimodo/common/stored.npz',
           'Geometry + nominal IK':a.run/'diagnostics_v2/000/kimodo/candidate/native_ik_direct.npz',
           'Refinement, pre-root':a.run/'diagnostics_v2/000/kimodo/candidate/pre_root.npz',
           'Refinement, post-root':a.run/'diagnostics_v2/000/kimodo/candidate/stored.npz'}
    motions={}
    for name,path in paths.items():
        with np.load(path) as z:motions[name]=Motion(z['positions'],list(z['names']),z['times'],'model_fk')
    valid=np.ones(len(d.times),bool)
    for m in motions.values():valid&=(d.times>=m.times[0])&(d.times<=m.times[-1])
    ts=d.times[valid];t=ts-ts[0];truth=points(sample(r,ts),True)
    pp={name:points(sample(m,ts)) for name,m in {'D':d,**motions}.items()}
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained');colors={'D':'#777777','Common':'#2b6cb0','Geometry + nominal IK':'#179286','Refinement, pre-root':'#d36b20','Refinement, post-root':'#9b44ab'}
    def line(q,a,b):return q[b]-q[a]
    def knee(q):return 180-angle(q['right_hip']-q['right_knee'],q['right_ankle']-q['right_knee'])
    for name,q in pp.items():
        color=colors[name]
        if name!='Refinement, post-root':
            axes[0,0].plot(t,angle(line(truth,'left_hip','right_hip'),line(q,'left_hip','right_hip')),label=name,color=color)
            axes[0,1].plot(t,knee(q),label=name,color=color)
            axes[1,0].plot(t,angle(line(truth,'right_ankle','right_foot'),line(q,'right_ankle','right_foot')),label=name,color=color)
        if name!='Geometry + nominal IK':
            axes[1,1].plot(t,np.linalg.norm(q['right_knee']-truth['right_knee'],axis=-1)*100,label=name,color=color)
            axes[2,0].plot(t,np.linalg.norm(q['pelvis_mid']-truth['pelvis_mid'],axis=-1)*100,label=name,color=color)
    axes[0,1].plot(t,knee(truth),label='Original R',color='black',ls='--',lw=1.5)
    titles=[('Hip-line direction error vs R','deg'),('Right geometric knee flexion','deg'),
            ('Right foot direction error vs R','deg'),('Right knee world-position error vs R','cm'),
            ('Hip-midpoint world-position error vs R','cm')]
    for axis,(title,unit) in zip(axes.flat,titles):
        axis.set_title(title);axis.set_ylabel(unit);axis.set_xlabel('Seconds from common D start');axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8);axes[1,1].legend(fontsize=8)
    end_error=np.linalg.norm(pp['Refinement, post-root']['pelvis_mid'][-1]-truth['pelvis_mid'][-1])*100
    if end_error>5:
        axes[2,0].annotate('Old probe: endpoint roundoff bug',xy=(t[-1],end_error),xytext=(1.5,end_error*.8),
                           arrowprops={'arrowstyle':'->','color':'#9b44ab'},fontsize=8,color='#9b44ab')
    common=json.loads((a.run/'common_v1/000/kimodo/receipt.json').read_text())['geometry']['after']
    candidate=json.loads((a.run/'candidate_probe_v1/000/kimodo/receipt.json').read_text())['geometry']['after']
    names=['shoulder_width','rest_hip_to_shoulder_midpoint','right_thigh','right_shank'];x=np.arange(len(names))
    axes[2,1].bar(x-.18,[common[n]*100 for n in names],.36,label='Common',color=colors['Common'])
    axes[2,1].bar(x+.18,[candidate[n]*100 for n in names],.36,label='Independent geometry',color=colors['Geometry + nominal IK'])
    axes[2,1].set_xticks(x,['Shoulder width','Rest trunk span','Thigh','Shank']);axes[2,1].set_ylabel('cm');axes[2,1].set_title('Fixed model dimensions');axes[2,1].legend(fontsize=8)
    fig.suptitle('000 / Kimodo: original-reference diagnosis\nActual D timestamps; full fixed window, independent native-frame/model-FK meanings',fontsize=13)
    fig.savefig(a.output/'000_kimodo_mechanism.png',dpi=160);fig.savefig(a.output/'000_kimodo_mechanism.svg')
    (a.output/'plot_receipt.json').write_text(json.dumps({'scope':'D-clock original-reference plot; no fitting or parameter selection',
        'D_frames':len(d.times),'plotted_frames':len(ts),'files':{k:str(v) for k,v in paths.items()},'reference':str(frozen/'references/000.npz')},indent=2)+'\n')


if __name__=='__main__':main()
