"""Verify preserved references and report conversion / fitting / endpoint errors separately."""
import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import angles,region
from motiogen.evalkit.k.config import digest
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs');OUT=BASE/'evalkit_k33_conversion_20261117';LOG=ROOT/'develop_logs/test_runs/20261117_k_conversion'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
freeze=read(OUT/'freeze.json');light=read(OUT/'light.json');assert len(light)==656
assert 'Ran 60 tests' in (LOG/'tests.log').read_text() and (LOG/'tests.log').read_text().rstrip().endswith('OK')
for p,h in freeze['protected'].items():assert digest(ROOT/p)==h,p
for p,h in freeze['reference_hashes'].items():assert digest(p)==h,p
cfg=ROOT/'config/evalkit_k33_20261117';cfg.mkdir(exist_ok=True)
methods={}
for m in sorted({r['method'] for r in light}):
    rows=[r for r in light if r['method']==m];groups=sorted({r['group_id'] for r in rows})
    balance=lambda values:float(np.mean([np.mean([v for r,v in zip(rows,values) if r['group_id']==g]) for g in groups]))
    old=[r['variants']['legacy']['trunk_incline']['p95_abs'] for r in rows];new=[r['variants']['topology_v2']['trunk_incline']['p95_abs'] for r in rows]
    methods[m]={'windows':len(rows),'participant_groups':len(groups),'old_trunk_p95_group_mean_deg':balance(old),'new_trunk_p95_group_mean_deg':balance(new),
        'source_vector_diagnostic_p95_group_mean_deg':balance([r['source_shape_diagnostic']['p95_abs'] for r in rows]),'worse_than_1deg':sum(b>a+1 for a,b in zip(old,new)),
        'mapping_variants':list({json.dumps(r['mapping']['mapping'],sort_keys=True) for r in rows})}
results=[];flags=[];position_flags=[]
for row in read(OUT/'exports.json'):
    path=OUT/'fits'/row['window_key']/row['method'];r=read(path/'result.json');assert r['status']=='computed',r
    assert digest(row['reference'])==r['identity']['reference']
    old=read(BASE/'evalkit_k31_full_20261116'/row['window_key']/row['method']/'result.json')
    with np.load(path/'comparison.npz') as z:t=z['times'];R=z['R'];D=z['D'];F=z['F:new']
    with np.load(BASE/'evalkit_k31_full_20261116'/row['window_key']/row['method']/'comparison.npz') as z:
        np.testing.assert_array_equal(t,z['times']);np.testing.assert_array_equal(R,z['R']);Fo=z['F:k31']
    a={k:angles(v) for k,v in [('R',R),('new',F),('old',Fo)]};regions={}
    for reg in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']:
        keys=[k for k in a['R'] if region(k)==reg];rec={}
        for v in ['old','new']:
            rec[v]={'pose_p95_deg':float(np.mean([np.percentile(abs(a[v][k]-a['R'][k]),95) for k in keys])),
                'velocity_p95_deg_s':float(np.percentile(np.abs(np.stack([np.gradient(a[v][k],t)-np.gradient(a['R'][k],t) for k in keys])),95)),
                'max_step_deg':float(np.max(np.abs(np.stack([np.diff(a[v][k]) for k in keys]))))}
        rec['coverage_old']=old['coverage'][reg]['angle'];rec['coverage_new']=r['coverage'][reg];regions[reg]=rec
        reasons=[];x,y=rec['old'],rec['new']
        if y['pose_p95_deg']>x['pose_p95_deg']+1:reasons.append('pose')
        if y['velocity_p95_deg_s']>max(x['velocity_p95_deg_s']+20,x['velocity_p95_deg_s']*1.2):reasons.append('velocity')
        if y['max_step_deg']>max(x['max_step_deg']+5,x['max_step_deg']*1.2):reasons.append('step')
        if rec['coverage_new']<rec['coverage_old']-.05:reasons.append('coverage')
        if reasons:flags.append({'method':r['method'],'window_key':r['window_key'],'region':reg,'reasons':reasons,**rec})
    for space in ['raw','normalized']:
        for key in r['scores'][space]['angles']:
            for field in ['valid','total']:assert r['scores'][space]['angles'][key]['end_to_end'][field]==old['scores'][space]['angles'][key]['end_to_end'][field]
        for joint in ['left_wrist','right_wrist','left_foot','right_foot']:
            x=old['scores'][space]['positions'][joint]['end_to_end']['world']['p95_abs'];y=r['scores'][space]['positions'][joint]['end_to_end']['world']['p95_abs']
            if y>x+.01:position_flags.append({'method':r['method'],'window_key':r['window_key'],'space':space,'joint':joint,'old_m':x,'new_m':y})
    results.append({**{k:row[k] for k in ['id','method','window_key','group_id','subset']},'regions':regions,
      'trunk':{'old':old['scores']['raw']['angles']['trunk_incline'],'new':r['scores']['raw']['angles']['trunk_incline']},
      'positions':{space:{j:{v:r['scores'][space]['positions'][j][v] for v in ['degradation','fit_deviation','end_to_end']} for j in ['left_wrist','right_wrist','left_foot','right_foot']} for space in ['raw','normalized']},
      'output':str(path)})
assert len(results)==16
write(cfg/'summary.json',{'methods':methods,'fits':results,'flags':flags,'position_flags':position_flags,'scope':'light scores are target-FK before serialization; sixteen fits score actual readback; original R; fixed windows and scoring counts; source-shape diagnostic not exported'})
write(cfg/'light.json',{'full_results':str(OUT/'light.json'),'sha256':digest(OUT/'light.json'),'records':[{**{k:r[k] for k in ['method','id','window_key','group_id','subset','mapping']},'trunk':{p:r['variants'][p]['trunk_incline'] for p in ['legacy','topology_v2']},'source_shape_diagnostic':r['source_shape_diagnostic']} for r in light]});write(cfg/'freeze.json',freeze)
write(cfg/'model_audit.json',read(LOG/'model_audit.json'));write(cfg/'shape_ablation.json',read(LOG/'shape_ablation.json'))
write(cfg/'lock_ablation.json',read(BASE/'evalkit_k33_locks_20261117/results.json'))
write(cfg/'conversion_policy.json',{'version':'topology_v2','default_compatibility':'retarget default remains legacy; explicitly select topology_v2 for corrected controlled inputs',
 'scope':'source/target spine node counts and monotone arc-length correspondence; fixed target rest geometry and native head/thorax frame contract retained',
 'source_given':'native hierarchy, positions, rotations and target asset offsets',
 'project_derived':'node correspondence by normalized cumulative chain length when counts differ; a declared correspondence hypothesis, not an upstream anatomical joint definition',
 'confirmed_fix':'four-node ARDY maps to four distinct source spine nodes; legacy repeated the proximal source node',
 'limitations':'fixed target shape changes pelvis-to-shoulder geometry; corrected structural mapping does not guarantee lower end-to-end error; ARDY worsening retained',
 'constraint_policy':'authored ranges and locks retained; no expanded-range or unlock diagnostic promoted; runtime receipt exposes ineffective locked variables; new explicit runtime config enforces model identity couplers after interpolation',
 'no_reference_leak':'only policy identifier is added to serialized provenance; reference-derived mapping/calibration arrays remain in experiment receipts'})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figdir=ROOT/'develop_logs/notes/20261117_k33';figdir.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(12,8))
for ax,(method,window) in zip(axes,[('motionlcm','031'),('ardy','051')]):
    with np.load(OUT/'fits'/window/method/'comparison.npz') as z:t=z['times'];cur={k:z[k] for k in ['R','D','F:new']}
    with np.load(BASE/'evalkit_k31_full_20261116'/window/method/'comparison.npz') as z:old={k:z[k] for k in ['D','F:k31']}
    for label,p in [('R original',cur['R']),('D legacy',old['D']),('D topology v2',cur['D']),('F legacy input',old['F:k31']),('F topology v2 input',cur['F:new'])]:ax.plot(t,angles(p)['trunk_incline'],label=label)
    ax.set_title(method+' / window '+window);ax.set_ylabel('Trunk incline (deg)');ax.set_xlabel('Fixed window time (s)');ax.legend(ncol=2);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(figdir/'trunk.png',dpi=150);plt.close(fig)
verification={'light_records':656,'fixed_windows':82,'fit_records':16,'scoring_windows_counts_unchanged':True,'original_reference_arrays_exactly_unchanged':True,
 'protected_files_unchanged':len(freeze['protected']),'reference_hashes_unchanged':len(freeze['reference_hashes']),'new_holdout_runs':0,
 'tests':60,'free_gib':shutil.disk_usage(OUT).free/1024**3,'figure_sha256':digest(figdir/'trunk.png'),
 'runtime_receipt_note':'constraint receipt, invalid-chain guards and identity-coupler synchronization added after fits; 17 postprocessed trajectories retain exact evaluated endpoints; full production recomputation verified separately; source snapshots retained',
 'source_snapshots':{str(p.relative_to(ROOT)):digest(p) for p in LOG.glob('*_experiment.py')}}
write(cfg/'verification.json',verification)
print(json.dumps(methods,indent=2));print('fits',len(results),'flags',flags,'position_flags',position_flags)
