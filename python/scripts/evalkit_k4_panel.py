"""Fix development ablation windows from completed common-baseline F-D scores."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    completion=json.loads((args.baseline/'completion.json').read_text())
    if completion['expected']!=656 or len(completion['results'])!=656:raise ValueError('Complete 656-path accounting required')
    methods=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
    expected={(f'{i:03d}',method) for i in range(82) for method in methods}
    if {(key,method) for key,method,_ in completion['results']}!=expected:
        raise ValueError('Missing or duplicated fixed window/format identities')
    rows=[];hashes={};failures=[]
    for key,method,status in completion['results']:
        if status!='completed':failures.append({'window':key,'method':method,'status':status});continue
        file=args.baseline/'scores'/f'{key}_{method}.json';data=json.loads(file.read_text())
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        metrics=data['comparisons']['F-D']['full']
        angles=[x['p95'] for g in ('directions_deg','joint_angles_deg') for x in metrics[g].values() if x['p95'] is not None]
        endpoints=[x['world']['p95'] for g in ('shoulders','hands','knees','feet') for x in metrics['positions_cm'][g].values() if x['world']['p95'] is not None]
        shoulders=[x['world']['p95'] for x in metrics['positions_cm']['shoulders'].values() if x['world']['p95'] is not None]
        rows.append({'window':key,'method':method,'worst_angle_p95':max(angles,default=-1),
                     'worst_endpoint_p95':max(endpoints,default=-1),'worst_shoulder_p95':max(shoulders,default=-1)})
    selected={}
    def add(row,reason):
        item=selected.setdefault((row['window'],row['method']),{**row,'reasons':[]})
        item['reasons'].append(reason)
    for method in methods:
        available=[r for r in rows if r['method']==method]
        for criterion in ('worst_angle_p95','worst_endpoint_p95'):
            if available:add(sorted(available,key=lambda r:(-r[criterion],r['window']))[0],criterion)
    for row in rows:
        if row['window']=='074' and row['method']=='kimodo':add(row,'fixed historical kneeling case')
    shoulder=[r for r in rows if r['window']=='017']
    if shoulder:add(sorted(shoulder,key=lambda r:(-r['worst_shoulder_p95'],methods.index(r['method'])))[0],
                    'fixed window 017; format selected by largest common-baseline shoulder F-D P95')
    result={'version':'k4-development-panel-1','baseline_identity':json.loads((args.baseline/'identity.json').read_text()),
            'selection':'Per format worst full-window F-D angle and endpoint; 074 Kimodo and 017 shoulder case; deduplicated; no R error used',
            'records':sorted(selected.values(),key=lambda r:(r['window'],methods.index(r['method']))),
            'failed_baseline_paths':failures,'score_hashes':hashes,'expected_paths':656,'new_holdout_uses':0}
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:raise ValueError('Fixed panel changed; use an explicit new version')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('fixed panel',len(selected),'baseline failures retained',len(failures))


if __name__=='__main__':main()
