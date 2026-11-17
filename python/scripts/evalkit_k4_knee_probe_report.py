"""Independent original-R acceptance and visualization of saved knee seed probes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle,normalize
from motiogen.evalkit.k.acceptance import compare
from evalkit_k4_knee_seed_probe import flexion,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--window',default='022');ap.add_argument('--method',default='kimodo')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((a.run/'receipt.json').read_text())
    freeze=json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())
    ref=Path(freeze['run'])/'references'/f'{a.window}.npz'
    with np.load(ref) as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
    source=Path(freeze['run'])/'inputs'/a.window/a.method/'candidate'
    d=load(source,a.method)
    rp,dp=points(sample(r,d.times),True),points(d)
    curves={'R':rp,'D':dp};summary={}
    for name in receipt['conditions']:
        file=a.run/'views'/name/a.window/a.method/'fk.npz'
        with np.load(file) as z:f=Motion(z['positions'],list(z['names']),z['times'],'model_fk')
        if not np.allclose(f.times,d.times,rtol=0,atol=1e-9):raise ValueError('Probe/D native clock mismatch')
        curves[name]=points(f)
        scores=json.loads((a.run/'scores'/name/f'{a.window}_{a.method}.json').read_text())
        summary[name]={'scores':scores['comparisons'],'original_clock':scores['original_clock']['comparisons'],
                       'independent_checks':{},'marker_report':{}}
        for side in ('left','right'):
            def values(p):return flexion(*[p[side+'_'+j] for j in ('hip','knee','ankle')])
            normal=normalize(rp,curves[name]);independent=abs(values(curves[name])-values(rp))
            observed=float(np.percentile(independent,95))
            registered=scores['comparisons']['F-R']['full']['joint_angles_deg'][side+'_knee']['p95']
            delta=abs(observed-registered)
            if delta>1e-6:raise ValueError('Independent atan2 and registered knee score disagree')
            invariant=float(np.max(abs(values(normal)-values(curves[name]))))
            if invariant>1e-6:raise ValueError('Length-only normalization changed geometric flexion')
            summary[name]['independent_checks'][side]={'atan2_P95_deg':observed,'score_difference_deg':delta,
                'length_normalization_max_change_deg':invariant,'R_range_deg':[float(values(rp).min()),float(values(rp).max())],
                'F_range_deg':[float(values(curves[name]).min()),float(values(curves[name]).max())]}
        marker=next((a.run/name/'ik').glob('*errors.sto'));lines=marker.read_text().splitlines()
        h=lines.index('endheader')+1;labels=lines[h].split();table=np.array([[float(v) for v in row.split()] for row in lines[h+1:] if row.strip()])
        for i,label in enumerate(labels[1:],1):summary[name]['marker_report'][label]={'median':float(np.median(table[:,i])),'p95':float(np.percentile(table[:,i],95))}
    base=json.loads((a.run/'scores/default_replay'/f'{a.window}_{a.method}.json').read_text())
    gates={name:compare(base,json.loads((a.run/'scores'/name/f'{a.window}_{a.method}.json').read_text()))
           for name in receipt['conditions'] if name!='default_replay'}
    (a.run/'initialization_acceptance.json').write_text(json.dumps(gates,indent=2)+'\n')
    result={'reference_sha256':sha(ref),'runtime_receipt_sha256':sha(a.run/'receipt.json'),'code_sha256':sha(__file__),
        'scope':'Independent R-only acceptance after probe processes exit; all frames, no source-jump exclusion; no model-anatomy accuracy claim',
        'frames':len(d.times),'conditions':summary}
    review=a.run/'independent_review.json'
    review.write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'plot_receipt.json').write_text(json.dumps({'review':str(review.resolve()),
        'review_sha256':sha(review),'code_sha256':sha(__file__),'reference_sha256':sha(ref),
        'window':a.window,'method':a.method,'alignment':'shared world coordinates; no per-motion alignment'},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,9));t=d.times-d.times[0]
    style={'R':('#222222','-'),'D':('#888888',':'),'default_replay':('#c54935','-'),'heading_knee':('#1767ae','-')}
    for c,side in enumerate(('left','right'),1):
        ax=fig.add_subplot(2,3,c)
        for name,(color,ls) in style.items():
            p=curves[name];ax.plot(t,flexion(*[p[side+'_'+j] for j in ('hip','knee','ankle')]),color=color,ls=ls,label=name)
        ax.set(title=side.title()+' geometric knee flexion',xlabel='Window time (s)',ylabel='Degrees (straight = 0)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=fig.add_subplot(2,3,3)
    for name in ('default_replay','heading_knee'):
        p=curves[name];ax.plot(t,angle(rp['right_hip']-rp['left_hip'],p['right_hip']-p['left_hip']),label=name,color=style[name][0])
    ax.set(title='Hip-line error against original R',xlabel='Window time (s)',ylabel='Degrees');ax.legend();ax.grid(alpha=.2)
    segments=[('left_hip','right_hip'),('left_shoulder','right_shoulder'),('pelvis_mid','shoulder_mid')]+[(s+'_'+x,s+'_'+y) for s in ('left','right') for x,y in [('hip','knee'),('knee','ankle'),('ankle','foot'),('shoulder','elbow'),('elbow','wrist')]]
    i=len(t)//2;center=rp['pelvis_mid'][i][[0,2,1]]
    for col,az in [(4,20),(5,110)]:
        ax=fig.add_subplot(2,3,col,projection='3d')
        for name in ('R','default_replay','heading_knee'):
            p=curves[name]
            for j,(x,y) in enumerate(segments):
                v=np.array([p[x][i],p[y][i]])[:,[0,2,1]];ax.plot(*v.T,color=style[name][0],alpha=.85,label=name if j==0 else None)
        for setter,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setter(c-.65,c+.65)
        ax.set_box_aspect((1,1,1));ax.view_init(18,az);ax.set(title=f'Mid-window, view {az} deg',xlabel='X (m)',ylabel='Z (m)',zlabel='Y up (m)');ax.legend(fontsize=8)
    ax=fig.add_subplot(2,3,6)
    for side,scan in receipt['knee_mapping'].items():
        ax.plot(scan['coordinate_deg'],scan['geometric_flexion_deg'],label=side)
        ax.axhline(scan['input_initial_flexion_deg'],ls=':',alpha=.5)
    ax.set(title='Fixed-model reachable knee geometry',xlabel='Model knee coordinate (deg)',ylabel='Geometric knee flexion (deg)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('022 Kimodo: same model and IK objective, input-derived initialization only',fontsize=15)
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(a.output/('knee_initialization.'+suffix),dpi=160)


if __name__=='__main__':main()
