"""Frozen actual-output K fits and independent F-D reports; no paired R exists."""
import argparse,hashlib,json,os,subprocess,sys,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_fit(args):
    sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
    import numpy as np
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.fidelity import score
    m=load(args.input,args.method,config=ROOT/'config/evalkit_k4_20261117/input.json');result={}
    for view in ('native','export'):
        path=args.views/view/'actual'/args.method/'fk.npz'
        with np.load(path) as z:data={k:z[k] for k in z.files}
        result[view]=score(m,data,data.get('head_relative_thorax'))
    args.output.write_text(json.dumps(dict(method=args.method,source_input_hashes=m.provenance['inputs'],views=result,
        scoring_implementation={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'utils/motiogen/evalkit/k/fidelity.py',ROOT/'utils/motiogen/evalkit/k/candidate_quality.py',ROOT/'utils/motiogen/evalkit/k/scoring.py']},
        scope='Actual generated output fidelity only; no original R or end-to-end accuracy',new_holdout_uses=0),indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('score')
    for name in ('input','views','output'):s.add_argument('--'+name,type=Path,required=True)
    s.add_argument('--method',required=True)
    r=sub.add_parser('run')
    for name in ('audit','code-root','fit-config','output'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if a.action=='score':return score_fit(a)
    if not 1<=a.workers<=3:raise ValueError('Explicit worker budget1..3 required')
    audit=json.loads(a.audit.read_text());rows=audit['rows']
    if len(rows)!=8 or any(x['status']!='read' for x in rows):raise ValueError('All eight audited actual inputs required')
    a.output.mkdir(parents=True,exist_ok=False)
    identity=dict(audit_file=str(a.audit),audit_sha256=sha(a.audit),fit_code_commit=subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),fit_config=str(a.fit_config),fit_config_sha256=sha(a.fit_config),runner_sha256=sha(__file__),workers=a.workers,new_holdout_uses=0)
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');shutil.copyfile(a.audit,a.output/'input_audit.json')
    (a.output/'progress.json').write_text(json.dumps(dict(expected=8,results=[]),indent=2)+'\n')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def run(row):
        method=row['method'];out=a.output/method;out.mkdir();fit=out/'fit';result=dict(method=method,status='failed')
        def call(script,options,log):
            with (out/(log+'.log')).open('w') as stream:subprocess.run([sys.executable,str(script),*map(str,options)],check=True,stdout=stream,stderr=subprocess.STDOUT,env=env)
        try:
            if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80GiB reserve required')
            for path,digest in row['input_hashes'].items():
                if sha(path)!=digest:raise ValueError('Actual input changed after audit')
            job=dict(input=row['input'],method=method,output=str(fit),mass_kg=75.,floor_m=0.,fit_config=str(a.fit_config));jp=out/'runtime_job.json';jp.write_text(json.dumps(job,indent=2)+'\n')
            call(a.code_root/'python/scripts/evalkit_k4_run.py',['--code-root',a.code_root,'--job',jp],'fit')
            receipt=json.loads((fit/'receipt.json').read_text())
            if receipt['key']['inputs']!=row['input_hashes']:raise ValueError('Fitted input identity differs from audit')
            call(a.code_root/'python/scripts/evalkit_k4_native_export_views.py',['--code-root',a.code_root,'--fit',fit,'--input',row['input'],'--method',method,'--window','actual','--output',out/'views'],'views')
            call(Path(__file__),['score','--input',row['input'],'--method',method,'--views',out/'views','--output',out/'fidelity.json'],'score')
            call(a.code_root/'python/scripts/evalkit_k4_export_audit.py',['--fit',fit,'--output',out/'actual_export'],'export')
            result.update(status='completed',selected=receipt.get('candidate_selection',{}).get('selected'),scope='F-D only')
        except Exception as error:
            import traceback
            result.update(error=str(error),traceback=traceback.format_exc())
        (out/'status.json').write_text(json.dumps(result,indent=2)+'\n');return result
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,row) for row in rows]):
            result=future.result();results.append(result);(a.output/'progress.json').write_text(json.dumps(dict(expected=8,results=results),indent=2)+'\n');print(result,flush=True)
    (a.output/'completion.json').write_text(json.dumps(dict(expected=8,results=results,promotion='none',scope='Actual-output fitting execution and F-D fidelity, no paired R'),indent=2)+'\n')


if __name__=='__main__':main()
