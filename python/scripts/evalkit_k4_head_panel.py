"""Head-chain model replay over the fixed development panel; body fits remain controls."""
import argparse,json,os,subprocess,sys,hashlib,shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','code-root','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--body-fits',type=Path)
    ap.add_argument('--panel',type=Path)
    ap.add_argument('--attachment-policy',choices=['registered_chest','input_fixed_offset'],default='registered_chest')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=a.code_root/'config/evalkit_k4_20261117/strict_v1'
    panel=json.loads((a.panel or config/'panel.json').read_text())
    (a.output/'panel.json').write_text(json.dumps(panel,indent=2)+'\n')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(a.code_root/'utils')+':'+str(a.code_root/'utils/motiogen')}
    results=[]
    identity={'commit':subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),
              'scope':'Declared source body fit unchanged, attached-chain replay only; full fit domain experiment separate','body_fits':str(a.body_fits or a.run/'common_v1'),'attachment_policy':a.attachment_policy,'workers':1,'new_holdout_uses':0}
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for row in panel['records']:
        if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
        w,m=row['window'],row['method'];job=json.loads((a.run/'jobs'/f'{w}_{m}.json').read_text());fit=a.output/'fits'/w/m
        folder=a.output/'tasks'/f'{w}_{m}';folder.mkdir(parents=True)
        def call(script,args,label):
            with (folder/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(a.code_root/'python/scripts'/script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result={**row,'status':'failed'}
        try:
            call('evalkit_k4_attach_head.py',['--fit',(a.body_fits or a.run/'common_v1')/w/m,'--input',job['input'],'--method',m,'--output',fit,'--attachment-policy',a.attachment_policy],'attach')
            call('evalkit_k4_views.py',['--fit',fit,'--input',job['input'],'--method',m,'--window',w,'--output',a.output/'views','--control'],'views')
            for root in ('before','after'):
                call('evalkit_k4_ablation_score.py',['--fits',a.output/'views'/root,'--baseline',a.run/'common_v1',
                     '--window',w,'--method',m,'--output',a.output/'scores'/root,'--strict'],'score_'+root)
            result['status']='completed'
        except Exception as error:result['error']=str(error)
        results.append(result);print(result,flush=True)
        (folder/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        (a.output/'progress.json').write_text(json.dumps({'results':results,'expected':len(panel['records'])},indent=2)+'\n')
    (a.output/'completion.json').write_text(json.dumps({'results':results,'expected':len(panel['records']),'promotion':'none'},indent=2)+'\n')


if __name__=='__main__':main()
