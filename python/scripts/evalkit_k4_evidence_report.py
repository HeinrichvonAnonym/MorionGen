"""Self-contained, per-record compressed evidence for all registered K4 paths."""
import argparse,base64,datetime,gzip,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    module=ROOT/'utils/motiogen/evalkit/reporting/evidence.py'
    spec=importlib.util.spec_from_file_location('read_evidence',module);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    protocol=json.loads((a.run/'protocol.json').read_text());rows=[];blobs={}
    preservation_path=a.run/('preservation_after.json' if (a.run/'preservation_after.json').exists() else 'preservation_before.json')
    preservation=json.loads(preservation_path.read_text())
    for record in protocol['records']:
        key=record['window']+'_'+record['method'];file=a.run/'tasks'/key/'result.json'
        row={k:record.get(k) for k in ('window','method','group_id','subset')};row.update(key=key,status='pending',passed=None)
        if file.exists():
            evidence=reader.read_record(a.run,record);row.update({k:evidence.get(k) for k in ('status','passed','selected','error')})
            raw=json.dumps(evidence,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            blobs[key]=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
        rows.append(row)
    payload=dict(rows=rows,blobs=blobs,title=protocol.get('report_title','K4'),snapshot=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest(),fit_code=protocol['fit_code_commit'],expected=len(rows),
        source_run=str(a.run),summary_page=Path(protocol['html_report']).name,
        presentation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),module,ROOT/'utils/motiogen/evalkit/reporting/evidence.html']},
        preservation=preservation,preservation_source=str(preservation_path),
        preservation_sha256=hashlib.sha256(preservation_path.read_bytes()).hexdigest(),
        new_holdout_uses=preservation.get('new_holdout_uses'),promotion='none')
    template=(ROOT/'utils/motiogen/evalkit/reporting/evidence.html').read_text()
    document=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/'))
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(document);temp.replace(a.output)
    print(json.dumps({'output':str(a.output),'registered':len(rows),'completed':sum(r['status']=='completed' for r in rows),'passed':sum(r['passed'] is True for r in rows)}))

if __name__=='__main__':main()
