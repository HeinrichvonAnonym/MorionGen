"""Immutable development failure snapshot with source and score provenance."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());root=Path(protocol['output'])
    records=[]
    for source in protocol['records']:
        key=source['window']+'_'+source['method'];folder=root/'tasks'/key
        row={k:source[k] for k in ('window','method','group_id','subset','recording_id','source_window_s')}
        result=folder/'result.json';summary=folder/'summary.json'
        if not result.exists():row['state']='pending'
        else:
            execution=json.loads(result.read_text());row['state']=execution['status']
            if execution['status']=='completed' and summary.exists():
                d=json.loads(summary.read_text());row.update(state='passed' if d['passed'] else 'metrics_failed',
                    selected=d['selected'],failures=d['failed_metrics'],clock=d['clock'],original_clock=d['original_clock'],
                    summary_file=str(summary),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                    export_audit=d['export_audit'])
                # Retain all three pairs; no subtraction of scalar percentiles.
                row['native_full_p95']={pair:{k:v.get('p95') for k,v in metrics.items()} for pair,metrics in d['metrics']['native']['full'].items()}
            else:row['error']=execution.get('error')
        records.append(row)
    counts=Counter(x['state'] for x in records)
    snapshot=dict(created_utc=datetime.now(timezone.utc).isoformat(),expected=len(records),counts=dict(counts),
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),fit_commit=protocol['fit_code_commit'],
        records=records,new_holdout_uses=0,scope='Paired formats of fixed development windows; pending and failures retained')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(snapshot,f,indent=2,allow_nan=False)
    print(json.dumps(dict(expected=len(records),counts=dict(counts))))


if __name__=='__main__':main()
