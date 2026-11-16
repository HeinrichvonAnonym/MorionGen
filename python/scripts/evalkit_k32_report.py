"""Score frozen branch-continuation development probes without changing masks."""
import json,sys,hashlib,shutil,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.config import digest,implementation_key
from motiogen.evalkit.k.scoring import angles
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
read=lambda p:json.loads(p.read_text())
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['branch','continue'],default='branch');args=ap.parse_args()
new=BASE/f'evalkit_k32_{args.phase}_20261116';old=BASE/'evalkit_k31_full_20261116'
freeze=read(new/'freeze.json')
rows=[];flags=[];position_flags=[]
for case in freeze['selection']:
    path=Path(case['window_key'])/case['method']/'result.json';a=read(old/path);b=read(new/path)
    assert b['status']=='computed';assert a['identity']==b['identity']
    assert digest(b['reference'])==b['identity']['reference']
    for space in ['raw','normalized']:
        for kind in ['angles','directions']:
            for metric in a['scores'][space][kind]:
                assert a['scores'][space][kind][metric]['degradation']==b['scores'][space][kind][metric]['degradation']
                for field in ['valid','total']:
                    assert a['scores'][space][kind][metric]['end_to_end'][field]==b['scores'][space][kind][metric]['end_to_end'][field]
        for metric in a['scores'][space]['positions']:
            assert a['scores'][space]['positions'][metric]['degradation']==b['scores'][space]['positions'][metric]['degradation']
    item={**case,'regions':{},'coverage':b['coverage'],'scores':b['scores'],'baseline_scores':a['scores']}
    for region in a['temporal']['regions']:
        before=a['temporal']['regions'][region]['new'];after=b['temporal']['regions'][region]['new']
        v={'K31':before,'candidate':after,'degradation':b['temporal']['regions'][region]['D'],
           'fit_deviation_p95_deg':b['temporal']['regions'][region]['fit_deviation_p95_deg']}
        item['regions'][region]=v
        reasons=[]
        if after['mean_metric_p95_deg']>before['mean_metric_p95_deg']+1:reasons.append('pose')
        if after['velocity_error_p95_deg_s']>max(before['velocity_error_p95_deg_s']*1.2,before['velocity_error_p95_deg_s']+20):reasons.append('velocity')
        if after['maximum_step_deg']>max(before['maximum_step_deg']*1.2,before['maximum_step_deg']+5):reasons.append('step')
        if after['input_direction_coverage']<before['input_direction_coverage']-.05:reasons.append('coverage')
        if reasons:flags.append({**case,'region':region,'reasons':reasons,**v})
    for space in ['raw','normalized']:
        for joint in ['left_wrist','right_wrist','left_foot','right_foot']:
            before=a['scores'][space]['positions'][joint]['end_to_end']['world']['p95_abs']
            after=b['scores'][space]['positions'][joint]['end_to_end']['world']['p95_abs']
            if after>before+.01:position_flags.append({**case,'space':space,'joint':joint,'K31_m':before,'candidate_m':after})
    rows.append(item)
protected=read(ROOT/'config/evalkit_k3_rescue_20261116/experiment.json')
files={**protected['protected']['files'],**protected['k3_sha256']}
for p,h in files.items():assert digest(ROOT/p)==h,p
out=ROOT/'develop_logs/test_runs/20261116_k32';out.mkdir(exist_ok=True)
(out/f'{args.phase}_summary.json').write_text(json.dumps({'selection':freeze,'records':rows,'flags':flags,'position_flags':position_flags,'protected_unchanged':len(files),'new_holdout_runs':0,'free_gib':shutil.disk_usage(BASE).free/1024**3},indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figdir=ROOT/'develop_logs/notes/20261116_k32';figdir.mkdir(exist_ok=True)
with np.load(new/'077/kimodo/comparison.npz') as z:t=z['times'];R=z['R'];D=z['D'];F=z['F:k31']
with np.load(old/'077/kimodo/comparison.npz') as z:O=z['F:k31']
fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
for label,p in [('R original',R),('D degraded',D),('K3.1',O),('branch continuation',F)]:
    angle=angles(p)['right_upper_arm_elevation'];axs[0].plot(t,angle,label=label);axs[1].plot(t,np.gradient(angle,t),label=label)
for folder,label in [(old,'K3.1'),(new,'branch continuation')]:
    with np.load(folder/'077/kimodo/kinematics.npz') as z:axs[2].plot(z['times'],z['coordinates'][:,list(z['labels']).index('arm_flex_r')],label=label)
for ax,title in zip(axs,['Right upper-arm elevation (deg)','Elevation velocity (deg/s)','Shoulder flexion coordinate (deg)']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
axs[-1].set_xlabel('Full-window time (s)');fig.tight_layout();fig.savefig(figdir/f'{args.phase}_wiping.png',dpi=150);plt.close(fig)
print('records',len(rows),'flags',len(flags),'position_flags',len(position_flags))
for f in flags:print(f['method'],f['window_key'],f['region'],f['reasons'])
for r in rows:
 print(r['method'],r['window_key'],[(k,round(v['K31']['mean_metric_p95_deg'],2),round(v['candidate']['mean_metric_p95_deg'],2)) for k,v in r['regions'].items() if k.endswith('arm')])
