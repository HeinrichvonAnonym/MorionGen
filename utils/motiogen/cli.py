import argparse
import json
import sys
from pathlib import Path

from .adapters import MissingResources,ResourceGate,Unsupported,create_adapter
from .audit import audit
from .config import ROOT,environment_root,load_config,scenario
from .metrics import evaluate
from .motion import load_motion
from .runtime import Run,event,gpu_lock,json_write,supervise


def environment_python(cfg,name):
    return environment_root(cfg,name)/'bin/python'


def main(argv=None):
    parser=argparse.ArgumentParser(description='A/B/E/H source-backed baseline experiments')
    parser.add_argument('action',choices=['audit','prepare','demo','evaluate','evaluate-native','train-smoke'])
    parser.add_argument('--baseline',required=True,choices=['all','motionlcm','omnicontrol','chois','ardy','kimodo'])
    parser.add_argument('--config',default=None)
    parser.add_argument('--scenario',default='official')
    parser.add_argument('--seed',type=int,default=None,help='Omit to run configured three seeds')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-part',choices=['source','assets','environment','all','text-access'],default='assets')
    parser.add_argument('--asset',action='append',help='Prepare only selected archive name, model, text or checkpoint; repeat for several')
    parser.add_argument('--input',help='Canonical motion NPZ for evaluate')
    parser.add_argument('--official-protocol',help='Provenance manifest for official distribution embeddings; independent of demo metrics')
    parser.add_argument('--benchmark',action='store_true',help='Separate cold run, warmup 3, measured 20')
    parser.add_argument('--variant',choices=['native','cached','fp16','offload'],default='native')
    parser.add_argument('--profile',choices=['smoke','stratified','full'],default='smoke',help='Native evaluation depth; full is preflight-only, stratified is Kimodo-only')
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    for spec in cfg['baselines'].values():
        spec['variant']=args.variant
    names=list(cfg['baselines']) if args.baseline=='all' else [args.baseline]
    if args.action=='evaluate' and not (args.input or args.official_protocol):
        parser.error('evaluate requires --input or --official-protocol')
    if args.official_protocol:
        cfg['_evaluation_protocol']=str(Path(args.official_protocol).resolve())
    cfg['_evaluation_profile']=args.profile
    if args.profile=='stratified':
        # 1,856 raw + 416 correction cases, followed by TMR and official
        # evaluation, is bounded but can approach six hours on a 16 GiB card.
        # Keep a two-hour watchdog margin; resume skips completed motions.
        cfg['resources']['timeout_seconds']=28800
    scene=scenario(args.scenario)
    if args.action=='train-smoke' and args.scenario=='official':
        scene=scenario('reach')
    if args.dry_run:
        print(json.dumps({'action':args.action,'baselines':names,'scenario':scene,'seeds':cfg['seeds'] if args.seed is None else [args.seed],
                          'prepare_part':args.prepare_part,'worker_environments':{name:str(environment_python(cfg,name)) for name in names},
                          'evaluation_profile':args.profile,
                          'training_policy':'fail closed unless reviewed evidence and bound fit current budgets','mutations':False},indent=2,ensure_ascii=False))
        return 0
    results=[]
    for name in names:
        repeated=args.action=='demo' or (args.action=='evaluate-native' and args.profile=='smoke')
        seeds=(cfg['seeds'] if args.seed is None else [args.seed]) if repeated else [args.seed or 0]
        for seed in seeds:
            run=Run(cfg,name,args.action,scene,seed,sys.argv)
            try:
                if args.action=='audit':
                    evidence=audit(cfg,name,run.path)
                    result=run.finish('success' if evidence['code_integrity'] else 'failed',audit_file=str(run.path/'audit.json'))
                elif args.action=='evaluate' and not args.official_protocol:
                    metric_policy=json.loads((run.path/'contracts.json').read_text())['metrics.json']['content']
                    result_data=evaluate(load_motion(args.input),scene,metric_policy)
                    json_write(run.path/'metrics.json',result_data)
                    result=run.finish('success',input=str(Path(args.input).resolve()))
                else:
                    is_prepare=args.action=='prepare'
                    host_only=is_prepare or (args.action=='evaluate-native' and args.profile=='full')
                    python=Path(sys.executable) if host_only else environment_python(cfg,name)
                    if not python.exists():
                        result=run.finish('missing_resources',f'Dedicated environment {cfg["baselines"][name]["environment"]} missing; use prepare --prepare-part environment')
                    else:
                        # Resource checks precede torch import or model construction.
                        if args.official_protocol:
                            from .official_eval import preflight
                            preflight(cfg,name,args.official_protocol)
                        elif args.action=='evaluate-native' and args.profile=='stratified':
                            if name!='kimodo':
                                raise Unsupported('stratified native evaluation is currently Kimodo-only')
                            from .native_eval import stratified_preflight
                            check=stratified_preflight(cfg)
                            if check['status']!='ready':
                                raise MissingResources(json.dumps(check['missing'],ensure_ascii=False))
                        elif not is_prepare and not (args.action=='evaluate-native' and args.profile=='full'):
                            adapter=create_adapter(cfg,name,run.path,scene,seed)
                            if args.action=='train-smoke':
                                from .training import preflight_training
                                preflight_training(adapter)
                            adapter.preflight()
                        command=[str(python),str(ROOT/'python/scripts/baseline_worker.py'),'--run-dir',str(run.path)]
                        if args.benchmark:
                            command.append('--benchmark')
                        if is_prepare:
                            command+=['--prepare-part',args.prepare_part]
                            for asset in args.asset or []:
                                command+=['--asset',asset]
                            from .downloads import policy
                            result=supervise(run,command,cfg,gpu=False,timeout=policy().get('download_timeout_seconds',3600)+3600)
                        elif args.official_protocol or (args.action=='evaluate-native' and args.profile=='full'):
                            result=supervise(run,command,cfg,gpu=False)
                        else:
                            with gpu_lock(cfg):
                                result=supervise(run,command,cfg,gpu=True)
                results.append(result)
            except (MissingResources,Unsupported,ResourceGate) as exc:
                status={MissingResources:'missing_resources',Unsupported:'unsupported',ResourceGate:'resource_gate'}[type(exc)]
                results.append(run.finish(status,str(exc)))
            except Exception as exc:
                results.append(run.finish('failed',f'{type(exc).__name__}: {exc}'))
    print(json.dumps(results,indent=2,ensure_ascii=False))
    # Blocked/unsupported is not a successful benchmark. Preserve results from other baselines.
    return 0 if all(x['status']=='success' for x in results) else 1
