"""Read-only localhost progress API and live enhancement of the frozen HTML report."""
import argparse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse,parse_qs


class Results:
    def __init__(self,run):
        self.run=Path(run);self.protocol=json.loads((self.run/'protocol.json').read_text())
        self.records={r['window']+'_'+r['method']:r for r in self.protocol['records']}

    def snapshot(self):
        rows=[];latest=0
        for key,record in self.records.items():
            task=self.run/'tasks'/key;result=task/'result.json'
            row=dict(window=record['window'],method=record['method'],status='pending',passed=None,running=False)
            if result.exists():
                data=json.loads(result.read_text());latest=max(latest,result.stat().st_mtime)
                row.update({k:data.get(k) for k in ('status','passed','selected','error')})
            elif (task/'state.json').exists():row['running']=True
            rows.append(row)
        formats={}
        for row in rows:
            c=formats.setdefault(row['method'],dict(total=0,evaluated=0,passed=0,execution_failed=0,pending=0,running=0))
            c['total']+=1;c['evaluated']+=row['status']=='completed';c['passed']+=row['passed'] is True
            c['execution_failed']+=row['status']=='failed';c['pending']+=row['status']=='pending';c['running']+=row['running']
        workers=self.protocol['workers']
        epochs=sorted((self.run/'execution_epochs').glob('*.json'))
        if epochs:workers=json.loads(epochs[-1].read_text())['effective_workers']
        return dict(rows=rows,by_format=formats,expected=len(rows),
                    **{k:sum(x[k] for x in formats.values()) for k in ('evaluated','passed','execution_failed','pending','running')},
                    worker_limit=workers,checked_at=datetime.now(timezone.utc).isoformat(),
                    latest_result_at=datetime.fromtimestamp(latest,timezone.utc).isoformat() if latest else None)

    def detail(self,key):
        if key not in self.records:raise KeyError(key)
        task=self.run/'tasks'/key
        result=json.loads((task/'result.json').read_text())
        if result['status']!='completed':return result
        s=json.loads((task/'summary.json').read_text())
        return dict(window=s['window'],method=s['method'],status='completed',passed=s['passed'],selected=s['selected'],
                    failed_metrics=s['failed_metrics'],coverage=s['acceptance']['native']['D_clock']['full_window_coverage'],
                    metrics={layer:{pair:{name:{k:value.get(k) for k in ('p95','n','denominator')}
                        for name,value in values.items()} for pair,values in pairs.items()}
                        for layer,pairs in s['metrics']['native'].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766);a=parser.parse_args()
    results=Results(a.run);report=Path(results.protocol['html_report']).resolve()
    script=Path(__file__).resolve().parents[2]/'utils/motiogen/evalkit/reporting/live.js'
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(a.directory),**kwargs)
        def respond(self,data,content_type):
            self.send_response(200);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            url=urlparse(self.path)
            try:
                if url.path=='/api/k4/progress':
                    return self.respond(json.dumps(results.snapshot(),ensure_ascii=False).encode(),'application/json; charset=utf-8')
                if url.path=='/api/k4/record':
                    key=parse_qs(url.query).get('key',[''])[0]
                    return self.respond(json.dumps(results.detail(key),ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
                if Path(self.translate_path(url.path)).resolve()==report:
                    document=report.read_text()
                    document=document.replace('</html>','<script>'+script.read_text()+'</script></html>')
                    return self.respond(document.encode(),'text/html; charset=utf-8')
            except (KeyError,FileNotFoundError):return self.send_error(404,'Record unavailable')
            except (ValueError,OSError):return self.send_error(503,'Result update in progress; retry later')
            return super().do_GET()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
