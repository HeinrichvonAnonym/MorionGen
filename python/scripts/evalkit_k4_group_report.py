"""Aggregate stored acceptance by format and registered participant-format group."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve each group report snapshot')
    module=ROOT/'utils/motiogen/evalkit/reporting/aggregation.py';spec=importlib.util.spec_from_file_location('aggregation',module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    protocol=json.loads((a.run/'protocol.json').read_text());rows=[];sources={}
    for r in protocol['records']:
        task=a.run/'tasks'/(r['window']+'_'+r['method']);row={**r,'status':'pending','passed':None}
        file=task/'result.json'
        if file.exists():
            raw=file.read_bytes();v=json.loads(raw);sources[str(file)]=hashlib.sha256(raw).hexdigest()
            if any(v.get(k)!=r[k] for k in ('window','method')):raise ValueError('Result identity changed')
            row.update(status=v['status'],passed=v.get('passed'))
            if v['status']=='completed':
                file=task/'summary.json';raw=file.read_bytes();s=json.loads(raw);sources[str(file)]=hashlib.sha256(raw).hexdigest()
                if s['passed']!=row['passed']:raise ValueError('Summary acceptance differs')
                row['acceptance']=s['acceptance']
        rows.append(row)
    report=m.summarize(rows);report.update(source_sha256=sources,protocol_sha256=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest(),implementation_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (module,Path(__file__))})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('expected_paths','completed_paths','passed_paths')} for k,v in report['by_format'].items()}))
if __name__=='__main__':main()
