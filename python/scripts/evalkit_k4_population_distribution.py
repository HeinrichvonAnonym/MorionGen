"""Compare all reconstructions with all reconstructions of complementary methods."""
import argparse,base64,hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from evalkit_k4_error_distribution import render,METHODS,LABELS
from motiogen.evalkit.reporting.error_distribution import complementary_methods


def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.evidence.read_text());snapshot=Path(d['snapshot']);raw=snapshot.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=d['snapshot_sha256']:raise ValueError('Snapshot mismatch')
 if 'All evaluated candidates' not in d['scope']:raise ValueError('Full reconstruction evidence required')
 snap=json.loads(raw);graph=complementary_methods(snapshot)
 plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 labels={'all':'全部候选的所有重构','complement':'互补方法的所有重构'};sections=[];summary={}
 for pop in labels:
  summary[pop]={}
  for method in METHODS:
   allowed=set(graph[method]['members'])
   chosen=[x for x in d['candidate_records'] if x['method']==method and (pop=='all' or x['candidate'] in allowed)]
   points=[x for x in d['points'] if x['method']==method and (pop=='all' or x['candidate'] in allowed)]
   windows=[r['window'] for r in snap['rows'] if r['fully_evaluated'] and r['method']==method]
   passed_routes={x['window'] for x in chosen if x['passed']}
   counts=dict(complete_routes=len(windows),routes_with_pass=len(passed_routes),zero_pass_windows=sorted(set(windows)-passed_routes),
     candidates=len(chosen),passed=sum(x['passed'] for x in chosen),failed=sum(not x['passed'] for x in chosen),points=len(points),
     methods=sorted({x['candidate'] for x in chosen}))
   out=a.output.with_name(a.output.stem+'_'+pop+a.output.suffix)
   stats,images=render(d,points,out,method,counts,population=labels[pop]);summary[pop][method]=dict(counts=counts,statistics=stats)
   section=f'<section data-scope="{pop}" data-method="{method}"><h2>{LABELS[method]} · {labels[pop]}</h2><p>完整验收 {len(windows)}/82 路；至少一个候选通过 {len(passed_routes)}/{len(windows)} 路；零通过窗口：{", ".join(counts["zero_pass_windows"]) or "无"}。{len(chosen)}个候选重构，{counts["passed"]}通过、{counts["failed"]}未通过。</p>'
   if pop=='complement':
    section+=f'<details><summary>互补依据：{len(allowed)}种方法、{len(graph[method]["pairs"])}组双向交叉关系</summary><p>方法属于至少一组互补关系即可纳入；纳入后，保留其在本格式全部完整窗口上的重构，包括未通过结果。</p><p>'+html.escape(', '.join(sorted(allowed)))+'</p><table><tr><th>方法A</th><th>方法B</th><th>A通过/B失败窗口</th><th>B通过/A失败窗口</th></tr>'
    for pair in graph[method]['pairs']:section+='<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in [pair['a'],pair['b'],', '.join(pair['a_only']),', '.join(pair['b_only'])])+'</tr>'
    section+='</table></details>'
   for kind in ['position','angle']:section+='<img alt="'+method+' '+pop+' '+kind+'" src="data:image/png;base64,'+base64.b64encode(images[kind].read_bytes()).decode()+'">'
   section+='<details><summary>数值摘要</summary><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre></details></section>';sections.append(section)
   print(pop,method,counts,flush=True)
 report=dict(snapshot_sha256=d['snapshot_sha256'],scope='Same fixed snapshot, all full-window export R-clock F-R statistics. Complementary sets are descriptive, post-hoc crossed verdicts; no best-of-R output selection.',graph=graph,populations=summary)
 a.evidence.with_name('distribution_population_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 body='<h1>全部重构与互补方法：误差分布对照</h1><p>沿用同一124路完整验收快照。全部结果纳入通过与未通过候选；互补口径纳入已证实互补的方法在所有完整窗口上的重构，也保留它们未通过的结果。</p><p>口径：原动作R时钟、实际导出F、F−R、完整窗口；位置为原始几何世界误差。每点＝窗口×候选×部位指标；左栏中位数，右栏P95。黑菱形/橙线为本行散点中位数/P95。位置与角度分图、八格式同尺度，两种口径也采用共同尺度。精确等价别名去重，欠观测项不补零。</p><p><strong>互补判据：</strong>同一格式、双方已验收的共同窗口中，既有A通过/B失败，也有B通过/A失败。属于任意一对的方法纳入集合，随后保留集合的全部重构。此处按验收结果事后识别方法互补关系，不是运行时选择器，也不是逐窗口按R挑最佳输出。红点表示候选整体未通过，可能因时序或覆盖失败，并不表示这个点必然超过5。</p>'
 body+='<p>各格式当前完成名单和候选组成不同，此图不用于直接排名。<a href="20261119_k4_error_distribution.html">原“仅通过候选”分布</a>仍保留。</p><nav><div>'+''.join(f'<button data-pop="{k}">{v}</button>' for k,v in labels.items())+'</div><div>'+''.join(f'<button data-format="{m}">{LABELS[m]}</button>' for m in METHODS)+'<button data-format="show-all">展开八格式</button></div></nav>'+''.join(sections)
 script="""<script>
let population='all',format='motionlcm';
function update(){document.querySelectorAll('section').forEach(s=>s.hidden=s.dataset.scope!==population||(format!=='show-all'&&s.dataset.method!==format));document.querySelectorAll('[data-pop]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.pop===population));document.querySelectorAll('[data-format]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.format===format));}
document.querySelectorAll('[data-pop]').forEach(b=>b.addEventListener('click',()=>{population=b.dataset.pop;update()}));document.querySelectorAll('[data-format]').forEach(b=>b.addEventListener('click',()=>{format=b.dataset.format;update()}));update();
</script>"""
 css='body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 24px;color:#172b43;background:#fafbfc}nav{position:sticky;top:0;background:#fafbfc;padding:10px 0;z-index:1}nav div{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}button{padding:9px 13px;border:1px solid #bacbdb;border-radius:5px;background:white;cursor:pointer}button[aria-pressed=true]{background:#164e78;color:white}img{width:100%;height:auto;margin:18px 0;background:white}pre{white-space:pre-wrap;font-size:12px}summary{cursor:pointer}table{font-size:12px;border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}details{overflow:auto} @media print{nav{display:none}section[hidden]{display:block}img{break-inside:avoid}}'
 a.output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4全部与互补重构分布</title><style>'+css+'</style>'+body+script+'</html>')

if __name__=='__main__':main()
