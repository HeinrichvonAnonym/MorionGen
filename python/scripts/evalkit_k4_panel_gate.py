"""Gate the remaining fixed-panel fits after six completed development sentinels.

This is development configuration selection, outside runtime fitting. It never
promotes defaults or opens the final holdout. Six accepted rows are reused
read-only instead of refitted; fourteen fixed rows remain.
"""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.strict_acceptance import assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','source','next-config','output','receipt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--wait-pid',type=int)
    p.add_argument('--review-only',action='store_true')
    a=p.parse_args();deadline=time.monotonic()+6*3600
    while not (a.source/'completion.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Source batch did not finish')
        if a.wait_pid:os.kill(a.wait_pid,0)
        time.sleep(5)
    if a.receipt.exists():raise ValueError('Gate receipt already exists')
    policy=json.loads((ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
    source=json.loads((a.source/'panel.json').read_text())['records'];next_panel=json.loads((a.next_config/'sentinels.json').read_text())
    original=json.loads((a.next_config/'full_panel.json').read_text())['records']
    key=lambda row:(row['window'],row['method'])
    if len(source)!=6 or len(next_panel['records'])!=14 or set(map(key,source))&set(map(key,next_panel['records'])):
        raise ValueError('Fixed six/remaining14 layout mismatch')
    if set(map(key,source+next_panel['records']))!=set(map(key,original)):
        raise ValueError('Combined panel differs from frozen20')
    identity=json.loads((a.source/'identity.json').read_text());selected=a.next_config/(a.condition+'.json')
    if not a.review_only:
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
        if identity['code_commit']!=commit or identity['config_sha256'].get(selected.name)!=hashlib.sha256(selected.read_bytes()).hexdigest():
            raise ValueError('Reused and future panel rows must have identical code/configuration')
    rows=[]
    for row in source:
        name=f"{row['window']}_{row['method']}.json";path=a.source/'scores'/a.condition/'after'/name
        if not path.exists():
            rows.append({**row,'status':'missing','score':str(path)});continue
        score=json.loads(path.read_text());baseline=json.loads((a.run/'common_v1/scores'/name).read_text())
        old=assess(score,baseline,policy);observed=assess(score,None,policy)
        rows.append({**row,'score':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'acceptance':old,'observed_input_control':observed,
            'status':'passed' if old['status']==observed['status']=='passed' else 'failed'})
    passed=all(row['status']=='passed' for row in rows)
    result={'gate_passed':passed,'records':rows,'scope':'Development only; no default promotion; source6 reused read-only',
            'new_holdout_uses':0,'action':'launch remaining14' if passed and not a.review_only else 'retain evidence without launching',
            'next_output':str(a.output)}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(result['action'],flush=True)
    if passed and not a.review_only:
        subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_optimize.py'),'--run',str(a.run),
            '--code-root',str(ROOT),'--config-root',str(a.next_config),'--output',str(a.output),'--workers','3'],check=True)


if __name__=='__main__':main()
