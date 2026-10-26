"""Materialize complete paired pilot blocks; no inference or downloads."""
import json
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT, load_config, environment_root
from motiogen.runtime import json_write
from motiogen.assets import find_asset

cfg=load_config()
parser=argparse.ArgumentParser()
parser.add_argument('--cohort',choices=['control','chois','dart','ardy','primal'],default='control')
parser.add_argument('--only-ids', nargs='*', help='Explicit subset for a new attempt; original receipts are preserved')
parser.add_argument('--output', type=Path)
args=parser.parse_args()
scene=json.loads((ROOT/'config/scenarios/reach.json').read_text())
jobs=[]
groups=[('motionlcm',[{'control_residual':x} for x in [0,1]]),
        ('omnicontrol',[{'realism':r,'spatial':s} for r in [0,1] for s in [0,1]])]
if args.cohort=='chois':
    groups=[('chois',[{'guidance':x} for x in [0,1]])]
    scene=json.loads((ROOT/'config/scenarios/carry.json').read_text())
if args.cohort=='ardy':
    groups=[('ardy',[{'history_frames':h,'correction':c} for h in [32,160] for c in [0,1]])]
    scene=json.loads((ROOT/'config/scenarios/online.json').read_text())
    scene['updates']=scene['updates'][:1]
    scene['updates'][0]['prompt']='A person walks backward.'
if args.cohort=='dart':
    groups=[]
    for condition in [0,1]:
        jobs.append({'id':f'dart_native_goal{condition}_seed20','method':'dart','seed':20,
            'scene':{'name':'native_goal_change','update_seconds':5,'duration_seconds':20},
            'method_spec':{'repo':'baselines/DART','environment':'motiogen-dart'},
            'command':[str(Path(cfg['data_root'])/'envs/motiogen-dart/bin/python'),
                str(ROOT/'python/scripts/dart_feedback_smoke.py'),'--native','--seed','20',
                '--frames','600','--goal-feedback',str(condition),'--output-dir','{run_dir}'],
            'required_outputs':['{run_dir}/motion.npz','{run_dir}/metrics.json','{run_dir}/intervention.json'],
            'require_protocol':False,'timeout_seconds':1800})
if args.cohort=='primal':
    groups=[]
    for initial in [1,2]:
        jobs.append({'id':f'primal_native_initial{initial}_seed20','method':'primal','seed':20,
            'scene':{'name':'native_continuation','initial_history':initial},
            'method_spec':{'repo':'baselines/PRIMAL','environment':'motiogen-dart'},
            'command':[str(Path(cfg['data_root'])/'envs/motiogen-dart/bin/python'),
                str(ROOT/'python/scripts/primal_feedback_smoke.py'),'--ground-ablation','--seed','20',
                '--frames','600','--initial-history',str(initial),'--output-dir','{run_dir}/factorial'],
            'required_outputs':['{run_dir}/factorial/timing.json','{run_dir}/factorial/initial_history.npz'],
            'require_protocol':False,'timeout_seconds':1800})
for method,conditions in groups:
    for index,condition in enumerate(conditions):
        jobs.append({'id':f'{method}_reach_seed20_condition{index}', 'method':method,
                     'seed':20,'scene':scene,'config':{'_intervention':condition},
                     'command':[str(environment_root(cfg,method)/'bin/python'),
                                str(ROOT/'python/scripts/baseline_worker.py'),'--run-dir','{run_dir}'],
                     'required_outputs':['{run_dir}/motion.npz','{run_dir}/metrics.json','{run_dir}/intervention.json'],
                     'timeout_seconds':1800})
revision=json.loads((ROOT/'config/survey_revision.json').read_text())
if args.cohort=='ardy':
    # Stage the 8B text model sequentially: two GPU lanes do not imply twice
    # the host RAM. Keep the reviewed 14 GiB worker bound and 5 GiB host reserve.
    for job in jobs:
        job['config']['resources']={**cfg['resources'],'ram_reserve_gib':5}
if args.only_ids is not None:
    unknown=set(args.only_ids)-{j['id'] for j in jobs}
    if unknown:
        raise ValueError(f'Unknown job IDs: {unknown}')
    jobs=[j for j in jobs if j['id'] in args.only_ids]
destination=args.output or ROOT/'develop_logs/test_runs/20261024_native_pilots'/f'{args.cohort}_plan.json'
identity_files=[]
evidence=json.loads((ROOT/'config/inference_evidence.json').read_text())
for method in {j['method'] for j in jobs}:
    entry=evidence.get(method,{})
    entry=entry.get('profiles',{}).get('full',entry)
    for item in entry.get('files',[]):
        if 'baseline_asset' in item:
            asset=find_asset(cfg,method,item['baseline_asset'])
            if asset is not None: identity_files.append(str(asset))
    if method=='dart':
        asset_base=Path(cfg['data_root'])/'outputs/assets/dart-official'
        identity_files.extend(str(p) for folder in ['policy_train','mld_denoiser','mvae']
                              for p in (asset_base/folder).rglob('*') if p.suffix in ['.pt','.pth','.yaml'])
        identity_files.append(str(asset_base/'data/stand.pkl'))
    if method=='primal':
        asset_base=Path(cfg['data_root'])/'outputs/assets/primal-official'
        identity_files.extend(str(p) for p in asset_base.rglob('*') if p.suffix in ['.ckpt','.yaml'])
json_write(destination,{'devices':[0] if args.cohort=='ardy' else [1] if args.cohort=='primal' else [0,1],
                        'deadline':revision['experiment_deadline_at'],'jobs':jobs,
                        'identity_files':sorted(set(identity_files))})
print(destination)
