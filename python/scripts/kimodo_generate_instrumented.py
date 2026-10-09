#!/usr/bin/env python
"""Batch-1 Kimodo benchmark generation with per-case timing and retained failures."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch

from benchmark.generate_eval import (_crop_output,_slice_output_at,copy_source_files,
    discover_example_folders)
from kimodo.constraints import load_constraints_lst
from kimodo.meta import parse_prompts_from_meta
from kimodo.model import load_model
from kimodo.tools import load_json,seed_everything

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.config import load_config
from motiogen.local_llm import precompute_encoder


def write_jsonl(path,row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='Kimodo-SOMA-RP-v1.1'); parser.add_argument('--postprocess',action='store_true')
    parser.add_argument('--limit',type=int); parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--text-cache')
    parser.add_argument('--timing-jsonl',required=True); parser.add_argument('--summary',required=True)
    args=parser.parse_args(); source=Path(args.benchmark).resolve(); output=Path(args.output).resolve()
    timing=Path(args.timing_jsonl); timing.parent.mkdir(parents=True,exist_ok=True)
    if args.overwrite and timing.exists(): timing.unlink()
    examples=discover_example_folders(source)
    if args.limit is not None: examples=examples[:args.limit]
    active=[item for item in examples if args.overwrite or not (output/item[1]/'motion.npz').is_file()]
    skipped=len(examples)-len(active)
    prompts=[]
    for src,_ in active:
        texts,_=parse_prompts_from_meta(load_json(src/'meta.json')); prompts.extend(texts)
    started=time.perf_counter(); text_started=time.perf_counter()
    if not active:
        summary={'status':'success','model':args.model,'postprocess':args.postprocess,'cases':len(examples),
            'success':0,'failed':0,'skipped':skipped,'text_precompute_seconds':0,'unique_prompts':0,
            'model_load_seconds':0,'elapsed_seconds':time.perf_counter()-started,'timing_jsonl':str(timing),'output':str(output)}
        Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(summary)); return 0
    frozen=precompute_encoder(load_config(),'kimodo',prompts,args.text_cache)
    text_seconds=time.perf_counter()-text_started
    model_started=time.perf_counter(); model,resolved=load_model(args.model,device='cuda:0',default_family='Kimodo',
        return_resolved_name=True,text_encoder=frozen)
    model_load=time.perf_counter()-model_started; process=psutil.Process(); success=failed=0
    for ordinal,(src,rel) in enumerate(active):
        dst=output/rel; motion_path=dst/'motion.npz'
        meta=load_json(src/'meta.json'); texts,durations=parse_prompts_from_meta(meta)
        text=texts[0]; frames=int(float(durations[0])*model.fps); cpath=src/'constraints.json'
        constraints=load_constraints_lst(str(cpath),model.skeleton,device=torch.device('cuda:0')) if cpath.is_file() else []
        copy_source_files(src,dst); case_ok=False
        for attempt in (1,2):
            seed=meta.get('seed');
            if seed is not None: seed_everything(seed)
            torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
            try:
                with torch.no_grad():
                    native=model([text],[frames],constraint_lst=[constraints],num_denoising_steps=meta.get('diffusion_steps',100),
                                 multi_prompt=False,post_processing=args.postprocess,return_numpy=True)
                elapsed=time.perf_counter()-begin; sample=_crop_output(_slice_output_at(native,0),frames)
                np.savez(motion_path,**sample); success+=1; case_ok=True
                write_jsonl(timing,{'case':str(rel),'status':'success','attempt':attempt,'ordinal':ordinal,
                    'seconds':elapsed,'duration_seconds':frames/model.fps,'rtf':elapsed/(frames/model.fps),
                    'frames':frames,'postprocess':args.postprocess,
                    'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_bytes':process.memory_info().rss})
                break
            except Exception as exc:
                write_jsonl(timing,{'case':str(rel),'status':'failed_attempt','attempt':attempt,'ordinal':ordinal,
                    'seconds':time.perf_counter()-begin,'reason':f'{type(exc).__name__}: {exc}',
                    'traceback':traceback.format_exc()[-4000:]})
                torch.cuda.empty_cache()
        if not case_ok: failed+=1
        print(json.dumps({'progress':ordinal+1,'total':len(active),'success':success,'failed':failed,'skipped':skipped}),flush=True)
    summary={'status':'success' if failed==0 else 'partial','model':resolved,'postprocess':args.postprocess,
        'cases':len(examples),'success':success,'failed':failed,'skipped':skipped,'text_precompute_seconds':text_seconds,
        'unique_prompts':len(set(prompts)),'model_load_seconds':model_load,
        'elapsed_seconds':time.perf_counter()-started,'timing_jsonl':str(timing),'output':str(output)}
    Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(summary,ensure_ascii=False))
    return 0 if failed==0 else 2


if __name__=='__main__': raise SystemExit(main())
