"""Replay a preregistered panel with at most three single-thread numerical workers."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if not 1<=a.workers<=3:raise ValueError('One to three workers required')
    protocol=json.loads(a.protocol.read_text());out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(a.protocol.read_text())
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT/'utils')+':'+str(ROOT/'utils/motiogen'))
    def run(row):
        w,m=row['window'],row['method'];dest=out/'replays'/w/m;dest.parent.mkdir(parents=True,exist_ok=True)
        receipt={'window':w,'method':m,'source_fit':row['fit'],'status':'running'}
        with (out/f'{w}_{m}.log').open('w') as log:
            try:
                if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80 GiB reserve reached')
                subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_rotation_probe.py'),'--fit',row['fit'],'--input',row['input'],'--output',str(dest),'--window',w,'--method',m,'--whole-body'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                for condition in protocol['conditions']:
                    subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_ablation_score.py'),'--fits',str(dest/'views'/condition),'--baseline',protocol['baseline'],'--output',str(out/'scores'/condition),'--window',w,'--method',m,'--strict'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                receipt['status']='completed'
            except Exception as error:receipt.update(status='failed',error=repr(error))
        return receipt
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(run,protocol['records']):
            completed.append(row);(out/'progress.json').write_text(json.dumps(completed,indent=2)+'\n');print(row,flush=True)
    (out/'completion.json').write_text(json.dumps(completed,indent=2)+'\n')
    if any(r['status']!='completed' for r in completed):sys.exit(1)


if __name__=='__main__':main()
