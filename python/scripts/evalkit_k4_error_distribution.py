"""Scatter plots of passing K4 candidate window medians and P95 values."""
import argparse,base64,collections,datetime,hashlib,html,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.reporting.error_distribution import collect,POSITION_GROUPS,ANGLE_GROUPS


def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--reuse-collected',action='store_true');a=p.parse_args()
 result=json.loads(a.evidence.read_text()) if a.reuse_collected else collect(a.snapshot)
 if result['snapshot_sha256']!=hashlib.sha256(a.snapshot.read_bytes()).hexdigest():raise ValueError('Collected snapshot mismatch')
 a.evidence.parent.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 methods=['motionlcm','omnicontrol','kimodo','ardy','chois','closd','dart','primal'];colors=dict(zip(methods,plt.get_cmap('tab10').colors[:8]));rng=np.random.default_rng(41)
 titlebase=f"{result['complete_routes']}路完整消融 · {result['routes_with_pass']}路有通过候选 · {result['passed_candidates_without_aliases']}个通过候选（去除精确等价别名）"
 summaries={};images={}
 for kind,label,unit,groupmap in [('position','位置','cm',POSITION_GROUPS),('angle','角度','°',ANGLE_GROUPS)]:
  pp=[x for x in result['points'] if x['kind']==kind];order=list(dict.fromkeys(groupmap.values()));order=[g for g in order if any(x['group']==g for x in pp)]
  fig,axes=plt.subplots(1,2,figsize=(15,max(6.5,len(order)*.53+2.5)),sharey=True)
  fig.subplots_adjust(left=.115,right=.98,bottom=.26,top=.83,wspace=.09);fig.suptitle(f'{label}端到端误差分布（F−R）\n{titlebase}',fontsize=15,y=.97)
  summaries[kind]={}
  for axis,stat,text in zip(axes,['median','p95'],['窗口内误差中位数','窗口内误差P95']):
   summaries[kind][stat]={}
   for yi,g in enumerate(order):
    rows=[x for x in pp if x['group']==g];values=np.array([x[stat] for x in rows]);med,p95=np.percentile(values,[50,95])
    summaries[kind][stat][g]=dict(points=len(rows),median=float(med),p95=float(p95),max=float(values.max()),fraction_le1=float(np.mean(values<=1)))
    for method in methods:
     vals=np.array([x[stat] for x in rows if x['method']==method]);axis.scatter(vals,yi+rng.uniform(-.26,.26,len(vals)),s=8,color=colors[method],alpha=.22,linewidths=0,rasterized=True)
    axis.scatter([med],[yi],marker='D',color='#111827',s=39,zorder=5,edgecolor='white',linewidth=.6)
    axis.plot([p95,p95],[yi-.27,yi+.27],color='#dc6b12',lw=2.3,zorder=5)
   axis.set_title(text,fontsize=13,pad=13);axis.set_yticks(range(len(order)),order);axis.set_ylim(len(order)-.5,-.6)
   axis.set_xscale('symlog',linthresh=.1,linscale=.6)
   maxval=max(x[stat] for x in pp);upper=max(5.8,maxval*1.13)
   ticks=[x for x in [0,.05,.1,.2,.5,1,2,5,10,20,50,100,180] if x<=upper]
   axis.set_xlim(0,upper);axis.set_xticks(ticks,[str(x) for x in ticks]);axis.tick_params(axis='x',labelsize=9)
   axis.axvline(5,color='#c02e36',ls='--',lw=1.3,zorder=3);axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True);axis.set_xlabel(f'误差（{unit}） · 0–0.1线性，其上为对数刻度')
  legend=[Line2D([],[],marker='o',ls='',color=colors[m],label=m,markersize=5) for m in methods]
  fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.075),ncol=8,frameon=False,fontsize=9)
  marks=[Line2D([],[],marker='D',ls='',color='#111827',label='本行散点的中位数'),Line2D([],[],marker='|',ls='',color='#dc6b12',markersize=12,markeredgewidth=2,label='本行散点的P95'),Line2D([],[],ls='--',color='#c02e36',label='5 cm / 5°工程验收线（针对窗口P95）')]
  fig.legend(handles=marks,loc='lower center',bbox_to_anchor=(.54,.12),ncol=3,frameon=False,fontsize=9)
  fig.text(.115,.035,'每点＝窗口×候选×一个部位指标。两侧指标分别计点；完整窗口保留来源跳变。',fontsize=9,color='#475569')
  fig.text(.115,.012,'标记是窗口统计值之间的分位数，并非汇总所有帧的分位数；仅展示通过候选，不代表总体通过率。',fontsize=9,color='#475569')
  out=a.output.with_name(a.output.stem+'_'+kind);fig.savefig(out.with_suffix('.png'),dpi=160);fig.savefig(out.with_suffix('.svg'));plt.close(fig);images[kind]=out.with_suffix('.png')
 result['summaries']=summaries;a.evidence.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 stamp=datetime.datetime.fromtimestamp(result['snapshot_time'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
 body=f'<h1>位置与角度误差：窗口统计分布</h1><p>{titlebase}</p><p>冻结快照：{stamp}（北京时间）。本页是固定快照。</p><p>口径：已完整验收路中的通过候选，原动作R时钟、实际导出F、F−R、完整窗口；位置为原始几何的世界位置误差。角度包括方向误差、肘膝几何角及可观测头胸相对旋转。欠观测项不补零；root和肩中点派生位置不纳入位置散点。</p><p>每点代表窗口×候选×一个部位指标的中位数或P95；黑菱形和橙短线分别为每行散点的中位数和P95。各候选、格式及左右侧为配对观测，图中不将其视为独立受试者。不按候选结果取最优，也不混池逐帧数据。完整窗口保留来源跳变，个别窗口P95可能超过连续层验收线；通过状态仍来自原冻结验收。</p>'
 for kind in ['position','angle']:
  body+='<img alt="'+kind+' distribution" src="data:image/png;base64,'+base64.b64encode(images[kind].read_bytes()).decode()+'">'
 body+='<details><summary>各部位数值摘要</summary><p>以下P95是图中窗口统计值的P95，不是逐帧总体P95。</p><pre>'+html.escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre></details>'
 a.output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4窗口误差分布</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#172b43;background:#fafbfc}h1{font-size:30px}p{max-width:1250px}img{width:100%;height:auto;background:white;margin:20px 0;border:1px solid #dce4ec}pre{white-space:pre-wrap;font-size:13px}summary{cursor:pointer} @media print{body{margin:0}img{break-inside:avoid}}</style>'+body+'</html>')
 print(json.dumps({k:result[k] for k in ['complete_routes','routes_with_pass','passed_candidates_without_aliases']},ensure_ascii=False));print('points',len(result['points']));print(json.dumps(summaries,ensure_ascii=False))

if __name__=='__main__':main()
