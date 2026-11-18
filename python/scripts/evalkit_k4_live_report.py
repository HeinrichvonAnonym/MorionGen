"""Read-only localhost progress API and live enhancement of the frozen HTML report."""
import argparse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import html
import re
from pathlib import Path
from urllib.parse import urlparse,parse_qs


class Results:
    def __init__(self,run,ablation_run=None):
        self.run=Path(run);self.protocol=json.loads((self.run/'protocol.json').read_text())
        self.records={r['window']+'_'+r['method']:r for r in self.protocol['records']}
        self.ablation_run=Path(ablation_run) if ablation_run else (self.run.parent/'full_candidate_ablation_20261118_v1' if 'budget_v3' in self.run.name else None)

    def ablation(self):
        path=self.ablation_run/'progress.json' if self.ablation_run else None
        missing=dict(available=False,reason='候选消融验收待建立；完成数、失败数与选择器比较均未知。',expected_routes=len(self.records))
        if path is None or not path.exists():return missing
        try:
            data=json.loads(path.read_text())
            required=('expected_routes','applicable_cells','independently_evaluated_cells','candidate_passed_cells',
                      'candidate_metric_failed_cells','execution_failed_cells','pending_cells','fully_evaluated_routes',
                      'all_candidates_passed_routes','selector_evaluated_routes','selector_passed_routes',
                      'selector_missed_passing_routes','selector_comparison_complete_routes','per_candidate','rows')
            if any(k not in data for k in required):return {**missing,'reason':'候选消融记录尚不完整；缺失统计保持未知。'}
            if (any(type(data[k]) is not int or data[k]<0 for k in required[:-2])
                    or not isinstance(data['per_candidate'],list) or not isinstance(data['rows'],list)):
                return {**missing,'reason':'候选消融统计格式尚不完整；当前统计未知。'}
            return {**data,'available':True,'source_file':str(path),'updated_at':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()}
        except (ValueError,OSError):return {**missing,'reason':'候选消融记录更新中；当前统计未知，稍后重试。'}

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
        progress_file=self.run/'progress.json'
        progress=json.loads(progress_file.read_text()) if progress_file.exists() else {}
        waiting=[]
        for item in progress.get('waiting_for',[]):
            path=Path(item);name=path.parent.name if path.suffix=='.json' else path.name
            if name not in waiting:waiting.append(name)
        return dict(rows=rows,by_format=formats,expected=len(rows),
                    **{k:sum(x[k] for x in formats.values()) for k in ('evaluated','passed','execution_failed','pending','running')},
                    worker_limit=workers,checked_at=datetime.now(timezone.utc).isoformat(),
                    run_name=self.run.name,recorded_state=progress.get('state'),stage=progress.get('stage'),waiting_on=waiting,
                    latest_result_at=datetime.fromtimestamp(latest,timezone.utc).isoformat() if latest else None,
                    ablation={k:v for k,v in self.ablation().items() if k!='rows'})

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


def ablation_panel(data):
    """Candidate ability and selector success have separate denominators."""
    esc=lambda value:html.escape(str(value))
    if not data.get('available'):
        body='<p>'+esc(data['reason'])+'</p>'
    else:
        n=data['expected_routes']
        body=(f"<p>所有适用候选独立验收完成 {data['fully_evaluated_routes']}/{n}；"
              f"所有候选均通过 {data['all_candidates_passed_routes']}/{n}（描述候选能力，不是完成条件）。</p>"
              f"<p>候选单元：独立验收 {data['independently_evaluated_cells']}/{data['applicable_cells']}；"
              f"通过 {data['candidate_passed_cells']}；指标未通过 {data['candidate_metric_failed_cells']}；"
              f"执行失败 {data['execution_failed_cells']}；待验 {data['pending_cells']}。</p>"
              f"<p>候选账本内的选择器输出验收：通过 {data['selector_passed_routes']}/{n}，已验 {data['selector_evaluated_routes']}/{n}；"
              f"候选能力与选择结果比较完成 {data['selector_comparison_complete_routes']}/{n}；"
              f"已有通过候选但选择器未通过 {data['selector_missed_passing_routes']}。</p>")
        body+='<div class="scroll"><table><tr><th>候选</th><th>通过／适用</th><th>已独立验收</th><th>指标未通过</th><th>执行失败</th><th>待验</th></tr>'
        for r in data['per_candidate']:
            value=lambda k:esc(r[k]) if k in r and r[k] is not None else '未知'
            body+='<tr>'+''.join('<td>'+x+'</td>' for x in [value('name'),value('passed')+'/'+value('applicable'),value('evaluated'),value('metric_failed'),value('execution_failed'),value('pending')])+'</tr>'
        body+='</table></div>'
        if data.get('updated_at'):body+='<p>候选账本更新时间：'+esc(data['updated_at'])+'；与主回归进度分别更新。</p>'
    return '<section id="k4-ablation"><h2>全部适用候选的独立验收</h2><p>24 列为已声明的完整策略及其导出变体；组合策略之间的比较不能直接得出单因素因果结论。完成条件是全部适用候选均有独立验收结果，不要求所有候选都通过。</p>'+body+'<p>适用性与选择仅使用输入 D；R 只用于独立验收。未验、执行失败、指标失败与不可观测分别保留。</p><p><a href="/docs/20261118_k4_candidate_ablation.html">候选消融逐路详情／离线快照</a></p></section>'


def ablation_document(data):
    title='K4 · 656 路完整候选消融'
    details=''
    def verdict(value):return '通过' if value is True else '未通过' if value is False else '未知'
    status_names={'pending':'待验','running':'计算中','completed':'已独立验收','inapplicable':'不适用','failed':'执行失败','execution_failed':'执行失败'}
    for row in data.get('rows',[]) if data.get('available') else []:
        label=' · '.join(str(row.get(k,'未知')) for k in ('window','method'))
        selector=row.get('selector',{})
        details+='<details><summary>'+html.escape(label)+' · '+('全部候选已验' if row.get('fully_evaluated') else '候选验收未完成')+'</summary>'
        details+='<p>选择器输出：'+verdict(selector.get('passed'))+'；所选候选：'+html.escape(str(selector.get('selected') or '尚未选定'))+'；候选比较：'+('已有通过候选但所选未通过' if row.get('missed_passing_candidate') is True else '已完成' if row.get('fully_evaluated') and selector.get('status')=='completed' else '尚未完成')+'。</p>'
        details+='<div class="scroll"><table><tr><th>候选</th><th>验收状态</th><th>独立结论</th><th>证据／原因</th></tr>'
        for name,cell in row.get('candidates',{}).items():
            state=cell.get('status','pending');evidence=''
            if state=='inapplicable':evidence='输入不满足声明的适用条件'
            elif state not in ('pending','running'):
                evidence='<details><summary>查看独立验收记录</summary><pre>'+html.escape(json.dumps(cell,ensure_ascii=False,indent=2))+'</pre></details>'
            details+='<tr><td>'+html.escape(name)+'</td><td>'+html.escape(status_names.get(state,state))+'</td><td>'+('不计适用分母' if state=='inapplicable' else verdict(cell.get('passed')) )+'</td><td>'+evidence+'</td></tr>'
        details+='</table></div></details>'
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>'+title+'</title><style>body{max-width:1160px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#203047;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #dbe2eb;padding:8px;text-align:left}details{padding:8px;margin:8px 0;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.scroll{overflow:auto}</style>'
            '<h1>'+title+'</h1><p>离线快照生成：'+html.escape(datetime.now(timezone.utc).isoformat())+'</p>'+ablation_panel(data)+details+'</html>')


def enhance_report(document,data,script):
    # The frozen writer may recreate the file; presentation is injected read-only.
    document=re.sub(r'(<h1>.*?</h1>\s*<p>)(?!选择器输出验收：)',r'\1选择器输出验收：',document,count=1,flags=re.S)
    document=document.replace('<th>通过／总数</th>','<th>选择器通过／总数</th>',1)
    document=document.replace('</table>','</table>'+ablation_panel(data),1)
    return document.replace('</html>','<script>'+script+'</script></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--ablation-run',type=Path)
    parser.add_argument('--ablation-report',type=Path)
    parser.add_argument('--snapshot-only',action='store_true');a=parser.parse_args()
    results=Results(a.run,a.ablation_run);report=Path(results.protocol['html_report']).resolve()
    ablation_report=a.ablation_report or a.directory/'docs/20261118_k4_candidate_ablation.html'
    def save_snapshot():
        ablation_report.parent.mkdir(parents=True,exist_ok=True)
        temporary=ablation_report.with_suffix('.tmp');temporary.write_text(ablation_document(results.ablation()));temporary.replace(ablation_report)
    save_snapshot()
    if a.snapshot_only:return
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
                if url.path=='/api/k4/ablation':
                    return self.respond(json.dumps(results.ablation(),ensure_ascii=False).encode(),'application/json; charset=utf-8')
                if url.path=='/api/k4/record':
                    key=parse_qs(url.query).get('key',[''])[0]
                    return self.respond(json.dumps(results.detail(key),ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
                if Path(self.translate_path(url.path)).resolve()==report:
                    document=enhance_report(report.read_text(),results.ablation(),script.read_text())
                    return self.respond(document.encode(),'text/html; charset=utf-8')
                if Path(self.translate_path(url.path)).resolve()==ablation_report.resolve():
                    return self.respond(ablation_document(results.ablation()).encode(),'text/html; charset=utf-8')
            except (KeyError,FileNotFoundError):return self.send_error(404,'Record unavailable')
            except (ValueError,OSError):return self.send_error(503,'Result update in progress; retry later')
            return super().do_GET()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
