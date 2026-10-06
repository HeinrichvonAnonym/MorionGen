"""Export a pinned upstream saved example, explicitly not a local inference result."""
import argparse
import ast
import json
import sys
import types
import importlib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'utils'))
from motiogen.adapters import joint_aliases
from motiogen.config import load_config
from motiogen.metrics import evaluate
from motiogen.motion import canonicalize,save_motion,load_motion
from motiogen.runtime import Run,checksum,json_write
from motiogen.visualize import render


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--example',default='04_ee_constraint')
    parser.add_argument('--constraint-fk',action='store_true',help='Use official CPU FK to overlay saved hand targets; requires torch/einops/scipy, no model loading')
    args=parser.parse_args()
    cfg=load_config(); source=ROOT/'baselines/kimodo/kimodo/assets/demo/examples/kimodo-soma-rp'/args.example
    # Read joint names as a literal AST value. No model/text/extension imports or allocations.
    tree=ast.parse((ROOT/'baselines/kimodo/kimodo/skeleton/definitions.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SOMASkeleton30')
    layout=next(ast.literal_eval(node.value) for node in cls.body if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='bone_order_names_with_parents' for x in node.targets))
    aliases=joint_aliases('somaskel30'); names=[aliases.get(x,x) for x,_ in layout]
    meta=json.loads((source/'meta.json').read_text())
    scene={'name':'kimodo_bundled_'+args.example,'kind':'saved_example','prompt':meta['text'],'duration_seconds':meta['duration'],'constraints':[],
           'scope':'Precomputed official repository sample; no local generation. Original constraint poses require native FK; hand-constraint error is unavailable in this export.'}
    if args.constraint_fk:
        import torch
        from scipy.spatial.transform import Rotation
        # Import the pinned pure FK modules without executing the package's
        # model/text loader __init__. No skeleton or model is substituted.
        package=ROOT/'baselines/kimodo/kimodo'
        for module,folder in (('kimodo',package),('kimodo.skeleton',package/'skeleton')):
            namespace=types.ModuleType(module); namespace.__path__=[str(folder)]
            sys.modules[module]=namespace
        fk=importlib.import_module('kimodo.skeleton.kinematics').fk
        constraint_items=json.loads((source/'constraints.json').read_text())
        joint_count=len(constraint_items[0]['local_joints_rot'][0])
        if joint_count not in (30,77):
            raise ValueError('Unsupported saved constraint skeleton')
        constraint_class=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==f'SOMASkeleton{joint_count}')
        constraint_layout=next(ast.literal_eval(node.value) for node in constraint_class.body if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='bone_order_names_with_parents' for x in node.targets))
        lookup={name:i for i,(name,_) in enumerate(constraint_layout)}
        skeleton=types.SimpleNamespace(neutral_joints=torch.load(package/f'assets/skeletons/somaskel{joint_count}/joints.p',map_location='cpu',weights_only=True).squeeze(),
                    joint_parents=torch.tensor([-1 if parent is None else lookup[parent] for _,parent in constraint_layout]),root_idx=lookup['Hips'])
        for item in constraint_items:
            joint={'left-hand':'LeftHand','right-hand':'RightHand'}.get(item['type'])
            if joint is None:
                continue
            vectors=np.asarray(item['local_joints_rot']); local=Rotation.from_rotvec(vectors.reshape(-1,3)).as_matrix().reshape(*vectors.shape[:2],3,3)
            rot,xyz,_=fk(torch.tensor(local,dtype=torch.float32),torch.tensor(item['root_positions'],dtype=torch.float32),skeleton)
            xyz,rot=canonicalize(xyz.numpy(),'Y-up',rot.numpy())
            for n,frame in enumerate(item['frame_indices']):
                scene['constraints'].append({'joint':aliases[joint],'time_seconds':frame/30,'position_m':xyz[n,lookup[joint]].tolist(),'rotation_matrix':rot[n,lookup[joint]].tolist()})
        scene['scope']='Official saved sample and official CPU FK hand-target conversion; no local generation. Finger/foot/root constraints excluded from these hand metrics; no postprocessing before/after pair.'
        scene['fk_source_sha256']=checksum(package/'skeleton/kinematics.py')
    run=Run(cfg,'kimodo','export-bundled',scene,meta['seed'],sys.argv)
    try:
        with np.load(source/'motion.npz',allow_pickle=False) as raw:
            xyz,rot=canonicalize(raw['posed_joints'],'Y-up',raw['global_rot_mats'])
            if xyz.shape[1]!=len(names):
                raise ValueError('Unexpected saved sample skeleton')
            metadata={'units':'m','axes':'right-handed Z-up','skeleton':'soma30','native_joint_names':[x for x,_ in layout],
                      'provenance':'official saved example, not local inference','source':str(source),'source_sha256':checksum(source/'motion.npz'),
                      'upstream_meta':meta,'postprocess':'upstream saved state; pre/post pair unavailable'}
            output=run.path/'motion.npz'; save_motion(output,xyz,names,30,metadata,rot,foot_contacts=raw['foot_contacts'])
        json_write(run.path/'metrics.json',evaluate(load_motion(output),scene))
        json_write(run.path/'original_constraints.json',json.loads((source/'constraints.json').read_text()))
        render(load_motion(output),scene,run.path/'motion.mp4')
        result=run.finish('success',scope=scene['scope'],motion=str(output))
    except Exception as exc:
        result=run.finish('failed',f'{type(exc).__name__}: {exc}')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='success' else 1


if __name__=='__main__':
    raise SystemExit(main())
