#!/usr/bin/env python
import argparse
import json
import os
import sys
import traceback
import importlib.metadata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.adapters import MissingResources,ResourceGate,Unsupported,create_adapter
from motiogen.config import repo_root
from motiogen.assets import prepare
from motiogen.audit import audit
from motiogen.metrics import evaluate
from motiogen.motion import load_motion
from motiogen.runtime import json_write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--prepare-part',default='assets')
    parser.add_argument('--asset',action='append')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args(); path=Path(args.run_dir)
    cfg=json.loads((path/'config.json').read_text()); manifest=json.loads((path/'manifest.json').read_text())
    metric_policy=json.loads((path/'contracts.json').read_text())['metrics.json']['content']
    name=manifest['baseline']; scene=manifest['scenario']; seed=manifest['seed']; action=manifest['action']
    result={}
    try:
        json_write(path/'environment_packages.json',{'python':sys.version,'packages':{x.metadata['Name']:x.version for x in importlib.metadata.distributions()}})
        if action=='prepare':
            result=prepare(cfg,name,path,args.prepare_part,args.asset)
        elif action=='evaluate-native' and cfg.get('_evaluation_profile')=='full':
            from motiogen.native_eval import full_preflight
            result=full_preflight(cfg,name,path)
        elif action=='evaluate-native' and cfg.get('_evaluation_profile')=='stratified':
            if name!='kimodo':
                raise Unsupported('stratified native evaluation is currently Kimodo-only')
            from motiogen.stratified import run_kimodo
            result=run_kimodo(cfg,path)
        elif action=='evaluate' and cfg.get('_evaluation_protocol'):
            from motiogen.official_eval import evaluate_official
            result=evaluate_official(cfg,name,cfg['_evaluation_protocol'],path,seed)
        else:
            import torch
            json_write(path/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda,
                                               'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
            if not torch.cuda.is_available():
                raise MissingResources('CUDA not available in the dedicated environment')
            repo=repo_root(cfg,name); os.chdir(repo); sys.path.insert(0,str(repo))
            adapter=create_adapter(cfg,name,path,scene,seed); adapter.preflight()
            if action=='train-smoke':
                from motiogen.training import preflight_training
                preflight_training(adapter)
            adapter.load()
            if action=='train-smoke':
                from motiogen.training import train_smoke
                result=train_smoke(adapter)
            else:
                output=adapter.run(args.benchmark)
                common=evaluate(load_motion(output),scene,metric_policy)
                json_write(path/'metrics.json',common)
                for extra in ('motion_pre_stitch.npz','motion_postprocess.npz'):
                    if (path/extra).exists():
                        json_write(path/(extra+'.metrics.json'),evaluate(load_motion(path/extra),scene,metric_policy))
                from motiogen.visualize import render
                render(load_motion(output),scene,path/'motion.mp4')
                timing=adapter.timing_summary()
                cold=[x for x in adapter.stages if x.get('cold',False) and x['stage']=='generation']
                measured=[x['seconds'] for x in adapter.stages if x['stage'] in ('generation','replan') and not x.get('warmup',False) and not x.get('cold',False)]
                total_generation=(sum(measured)/len(measured) if args.benchmark and measured else sum(x['seconds'] for x in adapter.stages if x['stage'] in ('generation','replan') and not x.get('warmup',False) and not x.get('benchmark',False)))
                duration=len(load_motion(output)['joints'])/adapter.fps
                result={'status':'success','motion':str(output),'timing':timing,'real_time_factor':total_generation/duration,
                        'cold_generation_seconds':cold[0]['seconds'] if cold else None,
                        'variant':cfg['baselines'][name].get('variant','native'),
                        'real_time_factor_scope':'mean steady generation / clip duration for benchmark; sum committed replans / total duration for online; generation includes nested text time',
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
                if action=='evaluate-native':
                    from motiogen.native_eval import smoke_result
                    result['evaluation']=smoke_result(cfg,name,output,path,common)
    except (MissingResources,Unsupported,ResourceGate) as exc:
        result={'status':{MissingResources:'missing_resources',Unsupported:'unsupported',ResourceGate:'resource_gate'}[type(exc)],'reason':str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result={'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
    json_write(path/'worker_result.json',result)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
