"""Bounded Kimodo native benchmark orchestration."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT,asset_root,repo_root
from .datasets import prepared_root,stratified_root
from .native_eval import stratified_preflight
from .runtime import json_write


def _environment(cfg):
    assets=asset_root(cfg); receipt=json.loads((assets/'text/llm2vec/local_base_receipt.json').read_text())
    env=dict(os.environ); repo=repo_root(cfg,'kimodo')
    env.update({'CHECKPOINT_DIR':str(assets/'kimodo'),'TEXT_ENCODERS_DIR':str(assets/'text/llm2vec'),
                'TEXT_ENCODER_MODE':'local','MORIONGEN_LLAMA_BASE_PATH':str(Path(receipt['files'][0]['path']).parent),
                'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
                'PYTHONPATH':os.pathsep.join([str(repo),str(ROOT/'utils'),env.get('PYTHONPATH','')])})
    return env


def _stage(name,command,cwd,env,run_dir,allow_partial=False):
    stdout=Path(run_dir)/(name+'.stdout.log'); stderr=Path(run_dir)/(name+'.stderr.log')
    started=time.time()
    with stdout.open('a') as out, stderr.open('a') as err:
        proc=subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=err)
    row={'stage':name,'command':[str(x) for x in command],'started_unix':started,'seconds':time.time()-started,
         'returncode':proc.returncode,'stdout':str(stdout),'stderr':str(stderr)}
    if proc.returncode and not allow_partial:
        raise RuntimeError(f'{name} failed with exit {proc.returncode}; see {stderr}')
    return row


def _postprocess_subset(testsuite,manifest,target):
    rows=json.loads(Path(manifest).read_text())['cases']; per_group={}
    for row in rows:
        if 'constraints_' not in row['group']: continue
        per_group.setdefault(row['group'],[]).append(row['case'])
    chosen=[case for group in sorted(per_group) for case in sorted(per_group[group])[:8]]
    if len(chosen)!=416: raise ValueError(f'expected 416 postprocess cases, got {len(chosen)}')
    for relative in chosen:
        source=Path(testsuite)/relative; destination=Path(target)/relative; destination.mkdir(parents=True,exist_ok=True)
        for name in ('meta.json','seed_motion.json','seed_constraints.json','constraints.json','gt_motion.npz'):
            if (source/name).is_file(): shutil.copy2(source/name,destination/name)
    return chosen


def run_kimodo(cfg,run_dir):
    check=stratified_preflight(cfg)
    if check['status']!='ready': return check
    root=stratified_root(cfg); suite=root/'testsuite'; raw=root/'generated_raw'; post_suite=root/'postprocess_testsuite'; post=root/'generated_postprocess'
    repo=repo_root(cfg,'kimodo'); python=Path(sys.executable); env=_environment(cfg); stages=[]
    stages.append(_stage('01_build_gt',[str(python),'benchmark/create_benchmark.py',str(suite),'--dataset',
        str(prepared_root(cfg)/'bones-seed-stratified-20261008/soma_uniform'),'--workers','8'],repo,env,run_dir))
    raw_timing=root/'raw_timing.jsonl'; raw_summary=root/'raw_generation_summary.json'
    text_cache=root/'text_embeddings.pt'
    stages.append(_stage('02_generate_raw',[str(python),str(ROOT/'python/scripts/kimodo_generate_instrumented.py'),
        '--benchmark',str(suite),'--output',str(raw),'--model','Kimodo-SOMA-RP-v1.1','--timing-jsonl',str(raw_timing),
        '--summary',str(raw_summary),'--text-cache',str(text_cache)],repo,env,run_dir,allow_partial=True))
    _postprocess_subset(suite,root/'selection_manifest.json',post_suite)
    post_timing=root/'postprocess_timing.jsonl'; post_summary=root/'postprocess_generation_summary.json'
    stages.append(_stage('03_generate_postprocess',[str(python),str(ROOT/'python/scripts/kimodo_generate_instrumented.py'),
        '--benchmark',str(post_suite),'--output',str(post),'--model','Kimodo-SOMA-RP-v1.1','--postprocess',
        '--timing-jsonl',str(post_timing),'--summary',str(post_summary),'--text-cache',str(text_cache)],repo,env,run_dir,allow_partial=True))
    stages.append(_stage('04_embed_raw',[str(python),str(ROOT/'python/scripts/kimodo_embed_instrumented.py'),str(raw),
        '--model','TMR-SOMA-RP-v1','--summary',str(root/'embedding_summary.json'),'--text-cache',str(text_cache)],repo,env,run_dir))
    stages.append(_stage('05_evaluate_raw',[str(python),'benchmark/evaluate_folder.py',str(raw)],repo,env,run_dir))
    stages.append(_stage('06_evaluate_postprocess',[str(python),'benchmark/evaluate_folder.py',str(post)],repo,env,run_dir))
    stages.append(_stage('07_parse_raw',[str(python),'benchmark/parse_folder.py',str(raw),'--output',str(root/'raw_summary_rows.json'),'--format','md'],repo,env,run_dir))
    analysis=Path(run_dir)/'supplemental_analysis'
    stages.append(_stage('08_supplemental',[str(python),str(ROOT/'python/scripts/analyze_kimodo_stratified.py'),
        '--raw',str(raw),'--timing',str(raw_timing),'--postprocess',str(post),'--postprocess-timing',str(post_timing),
        '--manifest',str(root/'selection_manifest.json'),'--output',str(analysis)],repo,env,run_dir,allow_partial=True))
    raw_result=json.loads(raw_summary.read_text()) if raw_summary.is_file() else {'status':'failed'}
    post_result=json.loads(post_summary.read_text()) if post_summary.is_file() else {'status':'failed'}
    result={'status':'success' if raw_result.get('failed')==0 and post_result.get('failed')==0 else 'partial',
            'profile':'stratified','baseline':'kimodo','selection':check['selection'],'execution_started':True,
            'raw_generation':raw_result,'postprocess_generation':post_result,'stages':stages,
            'raw_output':str(raw),'postprocess_output':str(post),'summary_rows':str(root/'raw_summary_rows.json'),
            'supplemental_analysis':str(analysis/'summary.json'),
            'scope':'1856-case deterministic stratified subset plus 416 paired postprocess ablation; not the 22474-case full benchmark'}
    json_write(Path(run_dir)/'native_evaluation.json',result)
    return result
