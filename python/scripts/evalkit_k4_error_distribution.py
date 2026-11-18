"""Scatter plots of passing K4 window statistics, pooled or separated by format."""
import argparse,base64,datetime,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.reporting.error_distribution import collect,POSITION_GROUPS,ANGLE_GROUPS
METHODS=['motionlcm','omnicontrol','kimodo','ardy','chois','closd','dart','primal']
LABELS=dict(zip(METHODS,['MotionLCM','OmniControl','Kimodo','ARDy','CHOIS','CLoSD','DART','PRIMAL']))
COLORS=dict(zip(METHODS,plt.get_cmap('tab10').colors[:8]))


def render(result,rows,output,method,counts,population='passing'):
    rng=np.random.default_rng(41);summaries={};images={}
    if population=='passing':
        title=f"{counts['complete_routes']}路完整消融 · {counts['routes_with_pass']}路有通过候选 · {counts['candidates']}个通过候选（去除精确等价别名）"
    else:
        title=f"{population} · {counts['complete_routes']}路 · {counts['candidates']}个重构（{counts['passed']}通过 / {counts['failed']}未通过）"
    for kind,label,unit,groupmap in [('position','位置','cm',POSITION_GROUPS),('angle','角度','°',ANGLE_GROUPS)]:
        pp=[x for x in rows if x['kind']==kind];order=list(dict.fromkeys(groupmap.values()))
        fig,axes=plt.subplots(1,2,figsize=(15,max(6.5,len(order)*.53+2.5)),sharey=True)
        fig.subplots_adjust(left=.115,right=.98,bottom=.26,top=.83,wspace=.09)
        fig.suptitle(f"{LABELS.get(method,'全部格式')} · {label}端到端误差分布（F−R）\n{title}",fontsize=15,y=.97)
        summaries[kind]={}
        # Both statistic panels and all formats use the same axis limits.
        upper=max(5.8,max(x[s] for x in result['points'] if x['kind']==kind for s in ('median','p95'))*1.13)
        for axis,stat,text in zip(axes,['median','p95'],['窗口内误差中位数','窗口内误差P95']):
            summaries[kind][stat]={}
            for yi,g in enumerate(order):
                group=[x for x in pp if x['group']==g];values=np.array([x[stat] for x in group])
                if not len(values):
                    summaries[kind][stat][g]=dict(points=0,median=None,p95=None)
                    axis.text(.16,yi,'独立旋转不可观测' if g=='头胸相对旋转' else '无可评分观测',color='#64748b',va='center',fontsize=10)
                    continue
                med,p95=np.percentile(values,[50,95])
                summaries[kind][stat][g]=dict(points=len(group),median=float(med),p95=float(p95),max=float(values.max()),fraction_le1=float(np.mean(values<=1)))
                for m in METHODS:
                    for passed in (True,False):
                        vals=np.array([x[stat] for x in group if x['method']==m and x.get('candidate_passed',True)==passed])
                        color=(COLORS[m] if population=='passing' else '#2474b7') if passed else '#d9364f'
                        axis.scatter(vals,yi+rng.uniform(-.26,.26,len(vals)),s=10,color=color,alpha=.35,linewidths=0,rasterized=True)
                axis.scatter([med],[yi],marker='D',color='#111827',s=39,zorder=5,edgecolor='white',linewidth=.6)
                axis.plot([p95,p95],[yi-.27,yi+.27],color='#dc6b12',lw=2.3,zorder=5)
            axis.set_title(text,fontsize=13,pad=13);axis.set_yticks(range(len(order)),order);axis.set_ylim(len(order)-.5,-.6)
            axis.set_xscale('symlog',linthresh=.1,linscale=.6)
            ticks=[x for x in [0,.05,.1,.2,.5,1,2,5,10,20,50,100,180] if x<=upper]
            axis.set_xlim(0,upper);axis.set_xticks(ticks,[str(x) for x in ticks]);axis.tick_params(axis='x',labelsize=9)
            axis.axvline(5,color='#c02e36',ls='--',lw=1.3,zorder=3);axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
            axis.set_xlabel(f'误差（{unit}） · 0–0.1线性，其上为对数刻度；八格式同尺度')
        legend=[Line2D([],[],marker='o',ls='',color=COLORS[m],label=LABELS[m],markersize=5) for m in METHODS if method=='all' or method==m]
        if population!='passing':legend=[Line2D([],[],marker='o',ls='',color='#2474b7',label='候选整体通过'),Line2D([],[],marker='o',ls='',color='#d9364f',label='候选整体未通过（含时序/覆盖失败）')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.075),ncol=8,frameon=False,fontsize=9)
        marks=[Line2D([],[],marker='D',ls='',color='#111827',label='本行散点的中位数'),Line2D([],[],marker='|',ls='',color='#dc6b12',markersize=12,markeredgewidth=2,label='本行散点的P95'),Line2D([],[],ls='--',color='#c02e36',label=f'5 {unit}工程验收线（针对窗口P95）')]
        fig.legend(handles=marks,loc='lower center',bbox_to_anchor=(.54,.12),ncol=3,frameon=False,fontsize=9)
        fig.text(.115,.035,'每点＝窗口×候选×一个部位指标。两侧分别计点；完整窗口保留来源跳变。',fontsize=9,color='#475569')
        footer='仅展示通过候选，不代表总体通过率。' if population=='passing' else '纳入所列方法全部重构；未按原动作挑选最佳候选。'
        fig.text(.115,.012,'标记是窗口统计值之间的分位数，并非逐帧总体分位数；'+footer,fontsize=9,color='#475569')
        suffix='' if method=='all' else '_'+method
        out=output.with_name(output.stem+suffix+'_'+kind)
        fig.savefig(out.with_suffix('.png'),dpi=160);fig.savefig(out.with_suffix('.svg'));plt.close(fig);images[kind]=out.with_suffix('.png')
    return summaries,images


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--reuse-collected',action='store_true');p.add_argument('--by-format',action='store_true');a=p.parse_args()
    result=json.loads(a.evidence.read_text()) if a.reuse_collected else collect(a.snapshot)
    if result['snapshot_sha256']!=hashlib.sha256(a.snapshot.read_bytes()).hexdigest():raise ValueError('Collected snapshot mismatch')
    a.evidence.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    snapshot=json.loads(a.snapshot.read_text());scopes=METHODS if a.by_format else ['all'];sections=[];summaries={}
    for method in scopes:
        rows=[x for x in result['points'] if method=='all' or x['method']==method]
        counts=dict(complete_routes=sum(r['fully_evaluated'] and (method=='all' or r['method']==method) for r in snapshot['rows']),
            routes_with_pass=len({(x['window'],x['method']) for x in rows}),candidates=len({(x['window'],x['method'],x['candidate']) for x in rows}),points=len(rows))
        stats,images=render(result,rows,a.output,method,counts);summaries[method]=dict(counts=counts,statistics=stats)
        section=f'<section id="{method}"><h2>{LABELS.get(method,"全部格式")}</h2><p>完整验收 {counts["complete_routes"]}/82 路；其中 {counts["routes_with_pass"]} 路有通过候选。本图包含 {counts["candidates"]} 个通过候选、{counts["points"]} 项部位统计。</p>' if a.by_format else '<section id="all">'
        for kind in ['position','angle']:
            section+='<img alt="'+method+' '+kind+' distribution" src="data:image/png;base64,'+base64.b64encode(images[kind].read_bytes()).decode()+'">'
        section+='<details><summary>本格式各部位数值摘要</summary><p>以下P95是图中窗口统计值的P95，不是逐帧总体P95。</p><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre></details></section>'
        sections.append(section);print(method,counts,flush=True)
    if a.by_format:
        a.evidence.with_name('distribution_by_format.json').write_text(json.dumps(dict(snapshot_sha256=result['snapshot_sha256'],scope=result['scope'],point_count=len(result['points']),formats=summaries),ensure_ascii=False,indent=2)+'\n')
    else:
        result['summaries']=summaries['all']['statistics'];a.evidence.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    stamp=datetime.datetime.fromtimestamp(result['snapshot_time'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    body=f'<h1>位置与角度误差：按退化格式查看</h1><p>同一份{result["complete_routes"]}路固定快照，{result["routes_with_pass"]}路有通过候选、{result["passed_candidates_without_aliases"]}个通过候选（去除精确等价别名）。快照：{stamp}（北京时间）。</p>'
    body+='<p>口径保持一致：完整验收路中的通过候选；原动作R时钟、实际导出F、F−R、完整窗口。位置为原始几何世界误差；角度包括骨段方向、肘膝几何角和可观测头胸相对旋转。欠观测项不补零。</p><p>每点＝窗口×候选×一个指标。左栏为窗口内中位数，右栏为窗口内P95；黑菱形和橙短线是本格式该行散点的中位数/P95。八格式使用相同刻度、范围和部位顺序。当前各格式完成名单不同，候选数也不同，不能直接据此排名。完整窗口保留来源跳变；各候选/格式/左右侧为配对观测，不是独立受试者。</p>'
    if a.by_format:
        body+='<nav aria-label="选择退化格式">'+''.join(f'<button type="button" data-method="{m}" aria-pressed="false">{LABELS[m]}</button>' for m in METHODS)+'<button type="button" data-method="show-all">展开全部</button></nav>'
    body+=''.join(sections)
    script="""<script>
const sections=[...document.querySelectorAll('section')];
function choose(method){sections.forEach(s=>s.hidden=method!=='show-all'&&s.id!==method);document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.method===method));}
document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>choose(b.dataset.method)));
if(document.querySelector('nav'))choose('motionlcm');
</script>"""
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#172b43;background:#fafbfc}h1{font-size:30px}p{max-width:1250px}img{width:100%;height:auto;background:white;margin:20px 0;border:1px solid #dce4ec}pre{white-space:pre-wrap;font-size:13px}summary{cursor:pointer}nav{position:sticky;top:0;padding:12px 0;background:#fafbfc;display:flex;gap:8px;flex-wrap:wrap;z-index:1}button{padding:9px 14px;border:1px solid #c8d4e1;border-radius:6px;background:white;color:#172b43;cursor:pointer}button[aria-pressed=true]{background:#164e78;color:white} @media print{body{margin:0}nav{display:none}section[hidden]{display:block}img{break-inside:avoid}}'
    a.output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4八格式窗口误差分布</title><style>'+style+'</style>'+body+script+'</html>')

if __name__=='__main__':main()
