"""Read-only paired check that D-R evidence stayed fixed across K versions.

This is a score-boundary audit, not a fit-accuracy test. Any changed reference,
clock, observation count or D-R value prevents a claim of directly paired K
improvement until the difference is explained and independently rescored.
"""
import argparse,hashlib,json
from pathlib import Path


def differences(a,b,path=''):
    if type(a)!=type(b):return [path+':type']
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [path+':keys']
        return [p for k in a for p in differences(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [path+':length']
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [path]


def boundary(data):
    result={k:data[k] for k in ['window','method','original_reference_file']}
    for clock,d in [('D_clock',data),('R_clock',data['original_clock'])]:
        result[clock]={k:d[k] for k in ['clock']}
        result[clock].update(positions_directions_and_timing=d['comparisons']['D-R'],
                            head_rotation=d['head_neck_rotation_comparisons']['D-R'])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-run','candidate-run','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--condition')
    a=ap.parse_args();root=a.candidate_run/'scores'
    if a.condition:root/=a.condition
    registered=json.loads((a.frozen_run/'protocol.json').read_text())['records']
    keys={r['window']+'_'+r['method'] for r in registered}
    sha=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    rows=[]
    for key in sorted(keys):
        task=a.candidate_run/'tasks'
        task=task/a.condition/key/'status.json' if a.condition else task/key/'result.json'
        if not task.exists() or json.loads(task.read_text()).get('status')!='completed':continue
        files={v:root/v/(key+'.json') for v in ['native','export']}
        if not all(p.exists() for p in files.values()):continue
        row=dict(key=key,views={},source_sha256={})
        for view,new in files.items():
            old=a.frozen_run/'scores'/view/new.name
            new_bytes=new.read_bytes();old_bytes=old.read_bytes()
            current=json.loads(new_bytes);previous=json.loads(old_bytes)
            changed=differences(boundary(previous),boundary(current))
            reference=Path(current['original_reference_file'])
            row['views'][view]=dict(exactly_equal=not changed,difference_count=len(changed),differences=changed)
            row['source_sha256'].update({str(old):hashlib.sha256(old_bytes).hexdigest(),
                                        str(new):hashlib.sha256(new_bytes).hexdigest(),str(reference):sha(reference)})
        row['passed']=all(v['exactly_equal'] for v in row['views'].values());rows.append(row)
    result=dict(scope=__doc__,expected=len(keys),checked=len(rows),equal=sum(r['passed'] for r in rows),
                pending=len(keys)-len(rows),records=rows,new_holdout_uses=0,
                implementation_sha256=sha(__file__),candidate_run=str(a.candidate_run),condition=a.condition)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['expected','checked','equal','pending']}))


if __name__=='__main__':main()
