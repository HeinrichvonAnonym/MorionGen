"""Original-R/native-FK/export attribution plots on the same physical clock."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.scoring import unit
from motiogen.degradation.metrics import points,angle
from evalkit_k4_ablation_score import at


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--window',required=True);p.add_argument('--method',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    scores=json.loads((a.run/'scores/native'/f'{a.window}_{a.method}.json').read_text());replay=a.run/'replays'/a.window/a.method
    def read(file,source=False):
        with np.load(file) as z:return Motion(z['positions'],list(z['names']),z['times'],'reference' if source else 'fk')
    native=read(replay/'views/native'/a.window/a.method/'fk.npz');ref=read(Path(scores['original_reference_file']),True)
    t=native.times;rp=points(at(ref,t),True);np_=points(native)
    freeze=json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text());artifacts=json.loads((Path(freeze['run'])/'artifact_verification.json').read_text())['records']
    record=next(r for r in artifacts if r['window']==a.window and r['method']==a.method)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
    palette={'native':'#334e68','linear':'#c55230','hip_slerp':'#c18b22','whole_so3':'#12816c'}
    labels={'native':'Native fitted FK','linear':'Euler-component interpolation','hip_slerp':'Hip-only SO(3)','whole_so3':'Whole eligible SO(3)'}
    for condition,color in palette.items():
        f=points(at(read(replay/'views'/condition/a.window/a.method/'fk.npz'),t))
        foot=angle(rp['left_foot']-rp['left_ankle'],f['left_foot']-f['left_ankle'])
        pelvis=angle(rp['right_hip']-rp['left_hip'],f['right_hip']-f['left_hip'])
        rate=np.linalg.norm(np.diff(unit(f['right_hip']-f['left_hip'])-unit(rp['right_hip']-rp['left_hip']),axis=0)/np.diff(t)[:,None],axis=-1)*180/np.pi
        lines=[(axes[0,0],t,foot),(axes[0,1],t,np.linalg.norm(f['left_foot']-np_['left_foot'],axis=-1)*100),(axes[1,0],t,pelvis),(axes[1,1],t[1:],rate)]
        for ax,clock,values in lines:ax.plot(clock-t[0],values,label=labels[condition],color=color,lw=1.5,alpha=.9)
    titles=['Left foot direction: exported FK vs original R','Left foot position: export minus native FK','Pelvis line direction: exported FK vs original R','Pelvis line direction-rate error vs original R']
    for ax,title,ylabel in zip(axes.flat,titles,['Degrees','cm','Degrees','Degrees / s']):
        ax.set_title(title,fontsize=10);ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        for event in record['source_jump_events']:
            ax.axvspan(max(t[0],event['start_s'])-t[0],min(t[-1],event['end_s'])-t[0],alpha=.1,color='gray')
    axes[0,0].legend(fontsize=8);axes[1,0].set_xlabel('Seconds from fixed-window start');axes[1,1].set_xlabel('Seconds from fixed-window start')
    fig.suptitle(f"{a.window} {a.method}: identical native fit / model / input-root policy",fontsize=14)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output.with_suffix('.svg'));fig.savefig(a.output.with_suffix('.png'),dpi=150);plt.close(fig)


if __name__=='__main__':main()
