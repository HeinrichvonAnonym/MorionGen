"""Original-motion acceptance, complete denominators and offline 656-path report."""
import argparse
import collections
import hashlib
import importlib.util
import html
import json
from pathlib import Path
import sys


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--code-root',type=Path,required=True)
    parser.add_argument('--record')
    a=parser.parse_args()
    acceptance_file=a.code_root/'utils/motiogen/evalkit/k/strict_acceptance.py'
    spec=importlib.util.spec_from_file_location('frozen_acceptance',acceptance_file)
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    assess=frozen.assess
    policy=json.loads((a.code_root/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text())
    protocol=json.loads((a.run/'protocol.json').read_text())
    if a.record:
        w,m=a.record.split('_',1);task=a.run/'tasks'/a.record
        view=json.loads((task/'views/receipt.json').read_text())
        fit_candidates=[Path(p)/w/m for p in protocol['reuse_fit_roots']]+[a.run/'fits'/w/m]
        fit=next(p for p in fit_candidates if (p/'receipt.json').exists())
        selection=json.loads((fit/'candidate_selection.json').read_text())
        scores={c:json.loads((a.run/'scores'/c/f'{w}_{m}.json').read_text()) for c in ('native','export')}
        acceptance={c:{'D_clock':assess(s,None,policy),'R_clock':assess(s['original_clock'],None,policy)} for c,s in scores.items()}
        audit=json.loads((task/'actual_export/audit.json').read_text())
        passed=all(v['status']=='passed' for block in acceptance.values() for v in block.values()) and audit['passed']
        # Independent score data stay separate from the D-only selector receipt.
        metrics={}
        for c,s in scores.items():
            metrics[c]={}
            for stratum in ('full','continuous','source_jump_tolerance'):
                metrics[c][stratum]={}
                for pair in ('D-R','F-D','F-R'):
                    b=s['comparisons'][pair][stratum];flat={**{'direction/'+k:v for k,v in b['directions_deg'].items()},
                        **{'angle/'+k:v for k,v in b['joint_angles_deg'].items()}}
                    for nodes in b['positions_cm'].values():
                        for n,geometries in nodes.items():
                            for g in ('world','pelvis_relative','common_length_world','common_length_pelvis_relative'):
                                flat['position/'+n+'/'+g]=geometries[g]
                    flat['rotation/head_relative_thorax']=s['head_neck_rotation_comparisons'][pair][stratum]
                    metrics[c][stratum][pair]=flat
        failed=[dict(condition=c,clock=k,**v) for c,b in acceptance.items() for k,g in b.items()
                for v in g['required']+g['temporal']+g['full_window_coverage'] if v['status'] not in ('passed','unobservable')]
        dump(task/'summary.json',dict(window=w,method=m,passed=passed,selected=selection['selected'],
            acceptance=acceptance,failed_metrics=failed,metrics=metrics,
            native_export_max_abs_position_m=view['native_export_max_abs_position_m'],export_audit=audit,
            candidate_selection_file=str(fit/'candidate_selection.json'),
            score_files={c:str(a.run/'scores'/c/f'{w}_{m}.json') for c in scores},
            clock=scores['native']['clock'],original_clock=scores['export']['original_clock']['clock'],
            input_boundary_missing_frames=scores['export']['original_clock']['input_boundary_missing_frames']))
        return
    rows=[];methods=sorted({r['method'] for r in protocol['records']})
    for entry in protocol['records']:
        w,m=entry['window'],entry['method'];task=a.run/'tasks'/f'{w}_{m}';result=task/'result.json'
        if result.exists():
            value=json.loads(result.read_text());row={k:value.get(k) for k in ('window','method','status','passed','fit','reused','error')}
            if value['status']=='completed':
                s=json.loads((task/'summary.json').read_text());row.update(selected=s['selected'],acceptance=s['acceptance'],
                    failed_metrics=s['failed_metrics'],metrics=s['metrics'],summary_file=str(task/'summary.json'))
        else:row=dict(window=w,method=m,status='pending',passed=None)
        row.update({k:entry.get(k) for k in ('group_id','subset','recording_id')})
        rows.append(row)
    counts={m:dict(total=sum(r['method']==m for r in rows),passed=sum(r['method']==m and r.get('passed') is True for r in rows),
        evaluated=sum(r['method']==m and r['status']=='completed' for r in rows),
        execution_failed=sum(r['method']==m and r['status']=='failed' for r in rows),
        pending=sum(r['method']==m and r['status']=='pending' for r in rows)) for m in methods}
    result=dict(expected=656,evaluated=sum(c['evaluated'] for c in counts.values()),passed=sum(c['passed'] for c in counts.values()),
                execution_failed=sum(c['execution_failed'] for c in counts.values()),pending=sum(c['pending'] for c in counts.values()),
                by_format=counts,rows=rows,new_holdout_uses=0,promotion='none',
                scope='Per-window targets on original R; frozen continuous/source-jump strata, full coverage and both clocks. Missing rows stay pending or execution_failed.')
    groups={};targets={m:{} for m in methods}
    for row in rows:
        key=row.get('group_id') or 'undeclared'
        group=groups.setdefault(key,dict(total=0,evaluated=0,passed=0,execution_failed=0,pending=0))
        group['total']+=1;group['evaluated']+=row['status']=='completed';group['passed']+=row.get('passed') is True
        group['execution_failed']+=row['status']=='failed';group['pending']+=row['status']=='pending'
        if row['status']=='completed':
            for metric in row['acceptance']['native']['D_clock']['required']:
                value=targets[row['method']].setdefault(metric['metric'],dict(passed=0,failed=0,unobservable=0,evaluated=0))
                value['evaluated']+=1
                value['unobservable' if metric['status']=='unobservable' else 'passed' if metric['status']=='passed' else 'failed']+=1
    result.update(by_participant_group=groups,native_targets_by_format=targets)
    dump(a.run/'results_656.json',result)
    # Compact HTML loads no external assets and does not embed bulky full scores.
    display=[]
    for r in rows:
        record={k:r.get(k) for k in ('window','method','status','passed','selected','error')}
        if r['status']=='completed':
            record['failed_metrics']=r['failed_metrics']
            record['metrics']={s:{pair:{name:{k:v.get(k) for k in ('p95','n','denominator')}
                for name,v in metrics.items()} for pair,metrics in pairs.items()}
                for s,pairs in r['metrics']['native'].items()}
            record['coverage']=r['acceptance']['native']['D_clock']['full_window_coverage']
        display.append(record)
    summary=f"已评分 {result['evaluated']}/656；通过 {result['passed']}/656；执行失败 {result['execution_failed']}；待完成 {result['pending']}。"
    table=''.join(f"<tr><td>{m}</td><td>{c['passed']}/{c['total']}</td><td>{c['evaluated']}</td><td>{c['execution_failed']}</td><td>{c['pending']}</td></tr>" for m,c in counts.items())
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>K4 · 656 路原动作验收</title>
<style>body{max-width:1160px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#203047;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{border:1px solid #dbe2eb;padding:7px;text-align:left}details{border:1px solid #dbe2eb;background:white;margin:8px 0;padding:10px}input,select{font:inherit;padding:8px;margin-right:12px}.pass{color:#087b60}.fail{color:#b94338}.pending{color:#718096}.scroll{overflow:auto}pre{white-space:pre-wrap;font-size:12px}@media print{input,select{display:none}details{break-inside:avoid}}</style>
<h1>K4：656 路原动作验收</h1><p>SUMMARY</p><p>每路使用协议声明的冻结候选集合及统一输入选择规则；适用性由输入字段和几何决定。R 仅进入独立评分。逐窗口按 5°／5 cm、时序及覆盖验收；来源跳变沿用冻结分层。默认配置尚未晋升。</p>
<p>下表始终保留每种格式 82 路分母。未完成、执行失败、指标未通过和不可观测分别表达。展开可查看原生 FK 的三类误差；完整导出与原动作时钟结果保存在结果 JSON。</p>
<table><tr><th>格式</th><th>通过／总数</th><th>已评分</th><th>执行失败</th><th>待完成</th></tr>TABLE</table>
<input id="query" placeholder="窗口编号或格式"><select id="state"><option value="all">全部</option><option value="passed">通过</option><option value="failed">未通过／执行失败</option><option value="pending">待完成</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="page"></span><div id="rows"></div>
<script>const rows=DATA; let page=0,total=0; const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(x){return x==null?'不可观测':Number(x).toFixed(3)}
function render(){let q=document.querySelector('#query').value.toLowerCase(),s=document.querySelector('#state').value;let filtered=rows.filter(r=>(r.window+' '+r.method).includes(q)&&(s==='all'||(s==='passed'&&r.passed===true)||(s==='failed'&&(r.passed===false||r.status==='failed'))||(s==='pending'&&r.status==='pending')));total=filtered.length;page=Math.min(page,Math.max(0,Math.ceil(total/24)-1));document.querySelector('#page').textContent=' 共 '+total+' 路，第 '+(page+1)+' 页';document.querySelector('#rows').innerHTML=filtered.slice(page*24,page*24+24).map(r=>{let state=r.status==='pending'?'待完成':r.status==='failed'?'执行失败':r.passed?'通过':'指标未通过';let content=r.metrics?'<p>所选整段候选：'+esc(r.selected)+'；下表为连续层 P95。位置单位 cm，方向／几何角／旋转单位 °。</p><div class="scroll"><table><tr><th>指标／几何</th><th>D−R</th><th>F−D</th><th>F−R</th><th>R/D 支持数</th><th>F 评分数</th></tr>'+Object.keys(r.metrics.continuous['D-R']).map(k=>'<tr><td>'+esc(k)+'</td>'+['D-R','F-D','F-R'].map(p=>'<td>'+fmt(r.metrics.continuous[p][k].p95)+'</td>').join('')+'<td>'+r.metrics.continuous['D-R'][k].n+'</td><td>'+r.metrics.continuous['F-R'][k].n+'</td></tr>').join('')+'</table></div><details><summary>失败与覆盖记录</summary><pre>'+esc(JSON.stringify({failures:r.failed_metrics,coverage:r.coverage},null,2))+'</pre></details><details><summary>完整窗口与来源跳变层</summary><pre>'+esc(JSON.stringify({full:r.metrics.full,source_jump:r.metrics.source_jump_tolerance},null,2))+'</pre></details>':esc(r.error||'结果尚未完成');return '<details><summary class="'+(r.passed?'pass':r.status==='pending'?'pending':'fail')+'">'+r.window+' · '+r.method+' · '+state+'</summary>'+content+'</details>'}).join('')}
document.querySelector('#query').oninput=()=>{page=0;render()};document.querySelector('#state').onchange=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);render()};document.querySelector('#next').onclick=()=>{page++;render()};render();</script></html>'''
    document=document.replace('SUMMARY',html.escape(summary)).replace('TABLE',table).replace('DATA',json.dumps(display,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    target=Path(protocol['html_report']);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.html.tmp');temp.write_text(document);temp.replace(target)
    note=Path(protocol['note_report']);note.parent.mkdir(parents=True,exist_ok=True)
    text='# K4：656 路固定开发回归\n\n'+summary+'\n\n本轮使用同一冻结实现与候选配置，以原动作 R 验收。拟合与选择只读取 D；降级损失、拟合偏差和端到端误差分别保存。逐窗口通过同时要求原生 FK、实际导出、原动作时钟、有效覆盖和时序检查通过。\n\n'
    text+='| 格式 | 通过／82 | 已评分 | 执行失败 | 待完成 |\n|---|---:|---:|---:|---:|\n'
    for m,c in counts.items():text+=f"| {m} | {c['passed']}/{c['total']} | {c['evaluated']} | {c['execution_failed']} | {c['pending']} |\n"
    text+='\n当前结果用于确定冻结版本的适用范围；默认配置保持原值。失败窗口及逐指标原因全部保留。最终留出新增使用为 0。\n\n'
    text+=f"[离线逐路报告]({target}) · [完整结果 JSON]({a.run/'results_656.json'}) · [运行协议]({a.run/'protocol.json'})\n"
    note.write_text(text)
    print(summary)


if __name__=='__main__':main()
