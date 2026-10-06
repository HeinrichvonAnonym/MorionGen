import ast
import sys
from pathlib import Path

from .assets import find_asset
from .config import ROOT, asset_root
from .runtime import budgets, command_output, json_write


def audit(cfg, baseline, run_dir):
    spec=cfg['baselines'][baseline]
    repo=ROOT/spec['repo']
    checks={name:command_output(cmd,repo) for name,cmd in {
        'connectivity':['git','fsck','--connectivity-only'],'submodules':['git','submodule','status'],
        'lfs':['git','lfs','ls-files'],'commit':['git','rev-parse','HEAD'],'working_tree':['git','status','--short']}.items()}
    py=Path(sys.executable).parents[2]/spec['environment']/'bin/python'
    environment=command_output([str(py),'-c','import json,sys,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{x.metadata["Name"]:x.version for x in m.distributions()}}))']) if py.exists() else {'returncode':None,'reason':'dedicated environment missing'}
    imports=set(); parse_failures=[]
    for file in repo.rglob('*.py'):
        if '.git' in file.parts:
            continue
        try:
            tree=ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    imports.update(x.name.split('.')[0] for x in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module and node.level==0:
                    imports.add(node.module.split('.')[0])
        except (SyntaxError,UnicodeDecodeError) as exc:
            parse_failures.append({'path':str(file.relative_to(repo)),'reason':str(exc)})
    training=spec['training_entry']
    result={'baseline':baseline,'repo_checks':checks,'code_integrity':all(checks[x]['returncode']==0 for x in ('connectivity','submodules')),
            'environment':environment,'imports':sorted(imports),'parse_failures':parse_failures,
            'checkpoint':str(find_asset(cfg,baseline,spec['model'])) if find_asset(cfg,baseline,spec['model']) else None,
            'training_entry':training,'training_entry_exists':bool(training and (repo/training).exists()),
            'training_verified':False,'lora_note':'LoRA reduces optimizer/gradient memory, not frozen weight or activation residency; no ready trainer assumed',
            'resources':budgets(cfg),'assets_root':str(asset_root(cfg)/baseline)}
    if baseline=='motionlcm':
        result['training_audit']={'mode':'ControlNet + trajectory encoder, native frozen VAE/text/denoiser','evidence':'train_motion_control.py:78-98','distillation_residency':['teacher','student','EMA target','VAE','text encoder'],'default_batch':128,'safe_smoke_candidate':True}
    elif baseline=='omnicontrol':
        result['training_audit']={'mode':'native FP32 diffusion training','evidence':'train/training_loop.py; model/cmdm.py','fixed_generation_frames':196,'default_batch':64,'frozen':'CLIP','lora_target_candidates':['hint encoder','control transformer attention/linear layers']}
    elif baseline=='chois':
        result['training_audit']={'mode':'full diffusion + EMA','evidence':'trainer_chois.py:Trainer/run_sample','default_batch':32,'inference_hazards':['optimizer object','EMA copy','dataset preloading','torch.load without map_location','SMPL-X geometry'],'lora_target_candidates':['diffusion transformer attention/linear layers']}
    else:
        result['training_audit']={'mode':'inference release; official training pipeline not found','backbone_may_be_differentiable':True,'missing':['training data pipeline','native loss protocol','optimizer/resume trainer'],'excluded':'full 8B text-encoder fine-tuning on 24GB','possible_route':'cache frozen embeddings + motion-backbone adapter/LoRA; requires new training implementation'}
    # A directory or partial weight file is not a usable checkpoint.
    checkpoint=find_asset(cfg,baseline,spec['model'])
    result['checkpoint_presence_scope']='path presence only; see separate asset gate'
    result['checkpoint_complete']=bool(checkpoint and (checkpoint.is_file() or (checkpoint/'source_receipt.json').is_file()))
    result['partial_assets']=[str(p.relative_to(asset_root(cfg))) for p in (asset_root(cfg)/baseline).rglob('*.part')]
    from .adapters import MissingResources,ResourceGate,Unsupported,create_adapter
    try:
        create_adapter(cfg,baseline,run_dir,{'name':'official','kind':'official','duration_seconds':5},0).preflight()
        result['asset_gate']={'assets_ready':True,'resource_review_passed':True}
    except MissingResources as exc:
        result['asset_gate']={'assets_ready':False,'reason':str(exc)}
    except ResourceGate as exc:
        result['asset_gate']={'assets_ready':True,'resource_review_passed':False,'reason':str(exc)}
    except Unsupported as exc:
        result['asset_gate']={'assets_ready':None,'reason':str(exc)}
    result['inference_verified']=False
    json_write(Path(run_dir)/'audit.json',result)
    return result
