"""Serial, frozen actual-output factor comparisons using existing K4 fit/scorers.

All results are F-D fidelity; this runner never receives a paired reference R.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('code-root','config-root','output'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();protocol_file=a.config_root/'protocol.json'
    p=json.loads(protocol_file.read_text())
    if not p['records'] or not p['conditions']:raise ValueError('Explicit records and factor conditions required')
    for name in p['conditions']:
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Condition must be a simple name')
    sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(protocol_file,a.output/'protocol.json')
    identity=dict(code_commit=subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),
                  config_sha256={f.name:sha(f) for f in a.config_root.glob('*.json')},
                  runner_sha256=sha(__file__),workers=1,new_holdout_uses=0,scope='Actual F-D only')
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    results=[];expected=len(p['records'])*len(p['conditions'])
    def publish(name):
        data=dict(expected=expected,results=results,promotion='none',scope='Actual F-D only')
        temp=a.output/(name+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(a.output/name)
    publish('progress.json')
    for row in p['records']:
        for condition in p['conditions']:
            method=row['method'];folder=a.output/condition/method;folder.mkdir(parents=True)
            fit=folder/'fit';status=dict(method=method,condition=condition,status='failed')
            def call(script,options,label):
                with (folder/(label+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(a.code_root/'python/scripts'/script),*map(str,options)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
            def check_input():
                if any(sha(f)!=h for f,h in row['input_hashes'].items()):raise ValueError('Actual input changed after registration')
            try:
                if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80GiB reserve required')
                check_input()
                if any(sha(a.config_root/f)!=h for f,h in identity['config_sha256'].items()):raise ValueError('Configuration changed')
                job=dict(input=row['input'],method=method,output=str(fit),mass_kg=75.,floor_m=0.,fit_config=str(a.config_root/(condition+'.json')))
                jp=folder/'runtime_job.json';jp.write_text(json.dumps(job,indent=2)+'\n')
                call('evalkit_k4_run.py',['--code-root',a.code_root,'--job',jp],'fit')
                receipt=json.loads((fit/'receipt.json').read_text())
                if receipt['key']['inputs']!=row['input_hashes']:raise ValueError('Fit used different input identity')
                call('evalkit_k4_native_export_views.py',['--code-root',a.code_root,'--fit',fit,'--input',row['input'],'--method',method,'--window','actual','--output',folder/'views'],'views')
                call('evalkit_k4_real_fidelity.py',['score','--input',row['input'],'--method',method,'--views',folder/'views','--output',folder/'fidelity.json'],'score')
                call('evalkit_k4_export_audit.py',['--fit',fit,'--output',folder/'actual_export'],'actual_export')
                check_input();status['status']='completed'
            except Exception as error:
                import traceback
                status.update(error=str(error),traceback=traceback.format_exc())
            (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            results.append(status);publish('progress.json');print(status,flush=True)
    publish('completion.json')


if __name__=='__main__':main()
