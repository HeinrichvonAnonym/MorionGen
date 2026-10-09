#!/usr/bin/env python
"""Offline Kimodo/TMR embedding with reviewed local Llama prompt precomputation."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmark.embed_folder import _load_posed_joints,discover_motion_folders
from kimodo.meta import parse_prompts_from_meta
from kimodo.tools import load_json

load_model_module=importlib.import_module('kimodo.model.load_model')

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.config import load_config
from motiogen.kimodo_compat import load_tmr_with_frozen_raw_encoder
from motiogen.local_llm import precompute_encoder


class FrozenEncoderModule(torch.nn.Module):
    """TMR registers its text encoder as a torch child module."""
    def __init__(self,frozen):
        super().__init__(); self.encoded=frozen.encoded; self._device=frozen._device

    def get_device(self): return torch.device(self._device)

    def forward(self,texts):
        scalar=isinstance(texts,str); keys=[texts] if scalar else list(texts)
        missing=[text for text in keys if text not in self.encoded]
        if missing: raise KeyError(f'Prompt was not precomputed: {missing}')
        vectors=torch.cat([self.encoded[text] for text in keys],dim=0).to(self._device)
        lengths=[1]*len(keys)
        return (vectors[0],lengths[0]) if scalar else (vectors,lengths)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path); parser.add_argument('--model',default='TMR-SOMA-RP-v1')
    parser.add_argument('--device',default=None); parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--text-cache')
    parser.add_argument('--summary',required=True)
    args=parser.parse_args(argv); folder=args.folder.resolve()
    dirs=discover_motion_folders(folder)
    if not dirs: raise SystemExit(f'No motion folders found under {folder}')
    def complete(sample_dir):
        required=['text_embedding.npy']
        if (sample_dir/'motion.npz').is_file(): required.append('motion_embedding.npy')
        if (sample_dir/'gt_motion.npz').is_file(): required.append('gt_motion_embedding.npy')
        return all((sample_dir/name).is_file() for name in required)
    active=dirs if args.overwrite else [sample_dir for sample_dir in dirs if not complete(sample_dir)]
    skipped=len(dirs)-len(active); started=time.perf_counter()
    if not active:
        summary={'schema_version':1,'status':'success','cases':len(dirs),'completed':0,'skipped':skipped,
                 'failed':0,'failures':[],'unique_prompts':0,'text_precompute_seconds':0,
                 'model_load_seconds':0,'elapsed_seconds':time.perf_counter()-started,'folder':str(folder)}
        Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(summary,ensure_ascii=False)); return 0
    texts=[]
    for sample_dir in active:
        parsed,_=parse_prompts_from_meta(load_json(sample_dir/'meta.json'))
        if len(parsed)!=1: raise SystemExit(f'Expected one prompt in {sample_dir}/meta.json')
        texts.append(parsed[0])
    text_started=time.perf_counter()
    frozen=FrozenEncoderModule(precompute_encoder(load_config(),'kimodo',texts,args.text_cache))
    text_seconds=time.perf_counter()-text_started
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_started=time.perf_counter()
    model=load_tmr_with_frozen_raw_encoder(load_model_module,args.model,device,frozen)
    model_seconds=time.perf_counter()-model_started
    completed=failed=0; failures=[]
    for sample_dir,text in zip(active,texts):
        try:
            for source_name,target_name in (('motion.npz','motion_embedding.npy'),('gt_motion.npz','gt_motion_embedding.npy')):
                source=sample_dir/source_name; target=sample_dir/target_name
                if source.is_file() and (args.overwrite or not target.is_file()):
                    posed=_load_posed_joints(source,device)
                    with torch.inference_mode(): embedding=model.encode_motion(posed,unit_vector=True)
                    np.save(target,embedding.detach().cpu().numpy())
            target=sample_dir/'text_embedding.npy'
            if args.overwrite or not target.is_file():
                with torch.inference_mode(): embedding=model.encode_raw_text([text],unit_vector=True)
                np.save(target,embedding.detach().cpu().numpy())
            completed+=1
        except Exception as exc:
            failures.append({'case':str(sample_dir.relative_to(folder)),'reason':f'{type(exc).__name__}: {exc}'})
            failed+=1
    summary={'schema_version':1,'status':'success' if failed==0 else 'partial','cases':len(dirs),
             'completed':completed,'skipped':skipped,'failed':failed,'failures':failures,'unique_prompts':len(set(texts)),
             'text_precompute_seconds':text_seconds,'model_load_seconds':model_seconds,
             'elapsed_seconds':time.perf_counter()-started,'folder':str(folder)}
    Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False)); return 0 if failed==0 else 2


if __name__=='__main__': raise SystemExit(main())
