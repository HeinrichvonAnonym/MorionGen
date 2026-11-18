"""Isolate input-only candidate choice from independent R acceptance.

This audits a diagnostic union of available candidates, not a prospective
uniform candidate bank. Missing and unfinished conditions stay explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def worker(job,code):
    sys.path[:0]=[str(code/'utils'),str(code/'utils/motiogen')]
    import numpy as np
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.candidate_quality import assess,choose
    d=json.loads(job.read_text())
    if set(d)!={'input','method','candidates','output','include_export_path','ranking_policy'}:raise ValueError('Unexpected runtime fields')
    motion=load(d['input'],d['method'],config=code/'config/evalkit_k4_20261117/input.json')
    qualities={};hashes={}
    for name,folder in d['candidates'].items():
        p=Path(folder);receipt=json.loads((p/'receipt.json').read_text())
        if receipt['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Candidate input mismatch')
        with np.load(p/'fit.npz') as z:fit={k:z[k] for k in z.files}
        with np.load(p/'head_neck.npz') as z:head=z['head_relative_thorax']
        qualities[name]=assess(motion,fit,head_rotation=head,include_export_path=d['include_export_path'],ranking_policy=d['ranking_policy'])
        hashes[name]={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['fit.npz','head_neck.npz','receipt.json']}
    dump(Path(d['output']),dict(selected=choose(qualities),qualities=qualities,source_sha256=hashes,
        scope='D-only native fidelity and optional fitted-anchor path diagnostic; no R in this child'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path);p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--frozen-run',type=Path)
    p.add_argument('--candidate-run',type=Path,action='append',default=[]);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.job:return worker(a.job,a.code_root)
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
    from motiogen.evalkit.k.strict_acceptance import assess
    policy=json.loads((a.code_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
    groups={}
    for run in a.candidate_run:
        panel=json.loads((run/'panel.json').read_text())['records'];protocol=json.loads((run/'protocol.json').read_text())
        for row in panel:
            w,m=row['window'],row['method'];key=w+'_'+m
            group=groups.setdefault(key,dict(window=w,method=m,candidates={},acceptance={},pending=[]))
            for condition in protocol['conditions']:
                name=run.name+'__'+condition;task=run/'tasks'/condition/key;status=task/'status.json'
                if not status.exists() or json.loads(status.read_text())['status']!='completed':
                    group['pending'].append(name);continue
                passed=json.loads((task/'actual_export/audit.json').read_text())['passed']
                for view in ('native','export'):
                    score=json.loads((run/'scores'/condition/view/(key+'.json')).read_text())
                    passed=passed and all(assess(s,None,policy)['status']=='passed' for s in (score,score['original_clock']))
                group['candidates'][name]=str(run/'fits'/condition/w/m);group['acceptance'][name]=bool(passed)
    results=[]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for key,group in sorted(groups.items()):
        frozen=a.frozen_run/'tasks'/key/'summary.json'
        if frozen.exists():
            old=json.loads(frozen.read_text());group['candidates']['frozen_selected']=str(Path(old['candidate_selection_file']).parent)
            group['acceptance']['frozen_selected']=old['passed']
        if not group['candidates']:
            results.append({**group,'status':'no_completed_candidate'});continue
        source=json.loads((a.run/'jobs'/(key+'.json')).read_text())
        choice={}
        for tag,dense,ranking in [('native_only',False,'flat_v1'),('with_path',True,'flat_v1'),('observed_then_path',True,'observed_then_path_v1')]:
            dest=a.output/key/tag;dest.mkdir(parents=True)
            job={k:source[k] for k in ('input','method')}
            job.update(candidates=group['candidates'],output=str(dest/'selection.json'),include_export_path=dense,ranking_policy=ranking)
            dump(dest/'runtime_job.json',job)
            with (dest/'runtime.log').open('w') as log:
                subprocess.run([sys.executable,__file__,'--job',str(dest/'runtime_job.json'),'--code-root',str(a.code_root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            selected=json.loads((dest/'selection.json').read_text())['selected']
            choice[tag]=dict(selected=selected,independent_R_passed=group['acceptance'][selected])
        any_pass=any(group['acceptance'].values())
        results.append({**group,'status':'reviewed','choice':choice,'any_candidate_R_passed':any_pass,
            'missed_passing_candidate':{k:any_pass and not v['independent_R_passed'] for k,v in choice.items()}})
        print(key,'any_pass',any_pass,'selected_pass',{k:v['independent_R_passed'] for k,v in choice.items()},flush=True)
        dump(a.output/'review.json',dict(records=results,expected=len(groups),reviewed=len(results),promotion='none',
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             selector_sha256=hashlib.sha256((a.code_root/'utils/motiogen/evalkit/k/candidate_quality.py').read_bytes()).hexdigest(),
             acceptance_sha256=hashlib.sha256((a.code_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_bytes()).hexdigest(),
             scope='Diagnostic union with incomplete/unequal candidate availability. Passing capability and D-only selection audited separately; not uniform-bank transfer qualification.'))


if __name__=='__main__':main()
