"""Independent acceptance of follow-up ablations; native/export and both clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--frozen-run',type=Path,required=True)
    parser.add_argument('--scoring-root',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.scoring_root/'utils'),str(args.scoring_root/'utils/motiogen')]
    from motiogen.evalkit.k.strict_acceptance import assess
    policy=json.loads((args.scoring_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
    panel=json.loads((args.run/'panel.json').read_text())['records']
    protocol=json.loads((args.run/'protocol.json').read_text())
    records=[]
    for entry in panel:
        w,m=entry['window'],entry['method'];key=w+'_'+m
        baseline=args.frozen_run/'tasks'/key/'summary.json'
        for condition in protocol['conditions']:
            task=args.run/'tasks'/condition/key
            state=json.loads((task/'status.json').read_text()) if (task/'status.json').exists() else {'status':'pending'}
            row=dict(window=w,method=m,condition=condition,status=state['status'])
            if state['status']=='completed':
                row['acceptance']={};row['metrics']={};row['source_sha256']={}
                for view in ('native','export'):
                    file=args.run/'scores'/condition/view/(key+'.json')
                    score=json.loads(file.read_text());row['source_sha256'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
                    row['acceptance'][view]={};row['metrics'][view]={}
                    for clock,data in [('D_clock',score),('R_clock',score['original_clock'])]:
                        row['acceptance'][view][clock]=assess(data,None,policy)
                        # Keep all three error pairs, raw/common lengths, full
                        # windows and source jump tolerance; no subtraction of P95s.
                        row['metrics'][view][clock]={k:data[k] for k in ['comparisons','head_neck_rotation_comparisons','clock']}
                row['export_audit']=json.loads((task/'actual_export/audit.json').read_text())
                row['passed']=row['export_audit']['passed'] and all(a['status']=='passed' for v in row['acceptance'].values() for a in v.values())
                row['failures']=[dict(view=v,clock=c,**x) for v,clocks in row['acceptance'].items() for c,a in clocks.items() for x in a['required']+a['temporal']+a['full_window_coverage'] if x['status'] not in ('passed','unobservable')]
                if baseline.exists():
                    old=json.loads(baseline.read_text());row['frozen_selected']=old['selected'];row['frozen_passed']=old['passed'];row['frozen_failures']=old['failed_metrics'];row['frozen_summary_file']=str(baseline)
            elif state['status']=='failed':row['error']=state.get('error')
            records.append(row)
    result={'records':records,'expected':len(records),'completed':sum(x['status']=='completed' for x in records),
            'passed':sum(x.get('passed',False) for x in records),'pending':sum(x['status']=='pending' for x in records),
            'execution_failed':sum(x['status']=='failed' for x in records),'promotion':'none',
            'policy_sha256':hashlib.sha256((args.scoring_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_bytes()).hexdigest(),
            'scope':'Full fixed development windows; R-only independent acceptance; exact frozen policy; F-D fidelity separate from F-R; no split changes'}
    dest=args.run/'review.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    for row in records:
        fs=[x for x in row.get('failures',[]) if x['view']=='native' and x['clock']=='D_clock']
        print(row['window'],row['method'],row['condition'],row['status'],row.get('passed'),[(x['metric'],round(x.get('p95',x.get('candidate',0)),3)) for x in fs])


if __name__=='__main__':main()
