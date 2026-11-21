"""Render the read-only TV failure inventory as a note and offline HTML."""
from collections import Counter
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
index=json.loads((HERE/'index.json').read_text())
data=json.loads(Path(index['evidence']).read_text())
names=dict(ardy='ARDy',chois='CHOIS',closd='CLoSD',dart='DART',kimodo='Kimodo',motionlcm='MotionLCM',omnicontrol='OmniControl',primal='PRIMAL')
activity=dict(Packaging='包装',Board_Connection='板件连接',Drilling='钻孔')
description={
 'T002':'左大腿、小腿与足方向；同时有左腿方向变化率超标。',
 'T008':'肩线、右上臂、右肘与左肩位置；右臂和肩线时序。',
 'T038':'双臂方向、双肘与右腕位置；双臂时序。',
 'T074':'右臂方向、右肘；部分格式还有躯干方向变化率。',
 'T075':'右臂、右肘及右腕的显著偏差；肩线与右臂时序，含前臂最大步长超标。',
 'T084':'右臂方向、右肘及右臂时序。',
 'T091':'右臂、右肘与右腕；原生拟合之外另有来源跳变层的输入偏差。',
 'T125':'仅时序：左上臂／前臂方向变化率。',
 'T226':'仅时序：肩线；四格式确认漏选，PRIMAL为提前接受失败且未访问项未知。',
 'T401':'仅姿态：右肘几何角的高误差尾部。',
 'T405':'左肘角度及左臂方向变化率。',
 'T420':'仅时序：右臂方向变化率；OmniControl在R时钟暴露，其余七格式已有原生D时钟失败。',
}


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def htmltable(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


windowrows=[]
for w in data['windows']:
    windowrows.append([w['window'],activity[w['activity']],Path(w['source']).name,
        '–'.join(f'{v:.3f}' for v in w['source_window_s']),
        '全部8种' if len(w['formats'])==8 else '、'.join(names[m] for m in w['formats']),description[w['window']]])
formatrows=[[names[m],f"{v['failed']}/{v['total']}",f"{v['failed']/v['total']:.2%}"] for m,v in data['format_counts'].items()]
bodyrows=[[k,data['body_route_counts'].get(k,0)] for k in ['头颈','肩/肩线','躯干','上臂/前臂','肘','腕','腿/膝','踝/足']]
representatives=[('T075','kimodo','joint_angles_deg/right_elbow','右肘角 °'),('T075','kimodo','positions_cm/right_wrist/world','右腕世界位置 cm'),('T038','kimodo','joint_angles_deg/right_elbow','右肘角 °'),('T401','kimodo','joint_angles_deg/right_elbow','右肘角 °'),('T405','kimodo','joint_angles_deg/left_elbow','左肘角 °')]
paired=[]
for w,m,k,label in representatives:
    x=next(x for x in data['records'] if x['window']==w and x['method']==m)
    s=x['scores']['export/R_clock']['continuous']
    paired.append([w+' '+names[m],label]+[f"{s[p][k]['p95']:.3f}" for p in ['D-R','F-D','F-R']]+[f"{s['F-R'][k]['median']:.3f}"])

sections=[
('结论与实验边界',[
 '本轮回答冻结D-only提前接受在电视测试上的实际表现。532条原始轨迹按中央5秒截取，短于5秒用全段，每条转换为八格式，共4256路；4168路通过，88路指标未通过，执行失败为0。88路集中在12条原始窗口。八格式是配对表示，不能当成八个独立动作样本。',
 '失败的主线是原生身体拟合及其时序保持。82路在原生FK的D采样点已出现超标；2路原生D时钟通过，但原生结果重采样至原动作R时钟后失败；4路原生D/R均通过，协调导出的R时钟失败。这是失效发生层次的定位，尚未把几何约束与求解局部解的贡献分开。',
 '选择器已确认4路漏选，全部位于T226；另82路已耗尽适用候选且没有独立通过项，2路提前接受失败而未访问项仍未知。已落盘候选具有补回这4路的能力；能否由通用D-only规则正确选中，仍需另行验证。现有证据尚未给出其他84路的通过方案。',
 '本次只读取既有评分、候选选择及文件回读审计，不重拟合、不改方法或阈值、不重选test结果、不新增留出运行、不触发全盘哈希扫描。Andy另行运行，继续保持14个worker；本报告的结论只属于电视测试。',
]),
('动作与格式分布',[
 '包装为19/424路未通过，涉及3/53条窗口；板件连接为69/2576路，涉及9/322条；钻孔为0/1256路，覆盖157条窗口。动作名称来自发布目录。具体持物、支撑及动作意图未通过视频核实，不据骨架文件名推断。',
 'MotionLCM与OmniControl各失败9路，其他格式失败11或12路。本次结果没有显示位置格式整体更差；各格式来自同一批源动作，失败高度聚集于少数窗口。下面时间为原BVH时间轴中的固定评分窗口。',
]),
('拟合偏差：局部高误差与持续偏差需要分开',[
 '64路含姿态或位置超标，其中56路还含时序失败；另24路只有时序超标，8路只有姿态超标。覆盖不足导致的失败为0。部位计数按路去重，各部位可重叠；头颈0表示这些可评分指标未触发失败，不向位置输入中不可观测的轴向转动授予准确度资格。',
 'T075最严重。Kimodo右肘端到端角误差中位数46.36°、P95 82.75°；D−R P95仅0.092°，F−D P95为82.74°。右腕F−R P95为20.14 cm，统一骨长后仍为19.58 cm。误差已经存在于原生拟合，并非仅由长度差或文件导出造成；但现有评分不能证明是限位、节点映射还是求解分支单独造成。',
 'T038与T401表现不同：Kimodo右肘中位数分别仅0.042°与0.055°，P95却为16.15°与7.84°，属于高误差尾部，不能描述为全段都拟合错误。T405左肘中位数3.10°、P95 8.89°，同时伴随左臂时序误差。',
 '下表统一采用实际导出、R时钟、冻结continuous评分层。保留D−R、F−D、F−R三组P95；范数与分位数不作加减归因。原始几何和统一骨长结果均保留在证据JSON。',
]),
('时序：原生轨迹、采样间路径与文件回读',[
 '80路有方向变化率或最大步长超标。T125、T226、T420的姿态／位置P95均通过，失败集中于时序。方向变化率约束衡量方向变化向量的误差，既可能来自局部跳变，也可能来自持续跟踪偏差，不能把80路全部叫作突跳。',
 '原生D时钟失败82路；新增原生R时钟失败为T226 PRIMAL与T420 OmniControl；新增实际导出R时钟失败为T226 CHOIS、CLoSD、DART、Kimodo。原生R结果由原生FK按R时间重采样，不是新测量；它能暴露采样间问题，但仅凭此无法独立确定原生求解和重采样各自贡献。',
 '88路所选结果的实际文件回读审计全部通过。因此没有证据把这批失败归因于坐标文件序列化或回读破坏。协调插值仍会改变采样间FK轨迹：T226四路就是路径变更损害原动作时钟精度的明确实例。',
 '冻结时序判据为：F−R方向变化率误差P95不超过D−R基线加10°/s；最大方向步长不超过对应D基线加5°。不能用统一绝对10°/s阈值判所有动作。T420 OmniControl右前臂在R时钟的D−R基线28.08°/s，F−R为42.16°/s，超过该路38.08°/s的门槛；其F−D仅4.44°/s，体现了D时钟代理与R时钟精度的差别。',
]),
('D-only：确认漏选与候选能力不足分开',[
 'T226的CHOIS、CLoSD、DART、Kimodo都访问过通过独立验收的pelvis_continuity，却提前接受了pelvis_continuity__coordinated。协调版本在输入侧的采样间路径代理更好，但实际导出在R时钟的肩线时序超标。选择器没有读取R；这里确认的是D代理排序没有保持R上的实际优劣。',
 '以CHOIS为例：原生候选在D-only的右前臂插值变化率代理为12.07°/s，未满足接受门槛；协调版本修正了该代理而获得接受。独立R时钟肩线F−R变化率P95却从14.25升至18.94°/s，门槛为5.078+10=15.078°/s。基线通过、协调后失败，这构成可核验的漏选证据。',
 '82路候选池已耗尽且无已访问通过候选：冻结候选集合尚未提供可通过的结果，单纯在这些既有候选中重新排序无法补救。T226 PRIMAL和T420 OmniControl为提前接受后失败，已访问候选中同样没有通过项，未访问部分的能力未知。本次不是完整test消融，不能报告全候选oracle通过率。',
]),
('来源跳变与归因边界',[
 '对88路，continuous层D时钟中各路最大骨段方向P95的最大值为0.0325°，最大位置P95为0.0554 cm；R时钟分别为1.100°和0.782 cm。D重采样带来的误差与方向变化率基线保留在评分中，不能将其一概宣称为零。',
 'T091 ARDy需特别保留：全窗口D时钟右腕D−R最大7.47 cm，位于已冻结的来源跳变宽容层；continuous层最大0.0359 cm、P95 0.0351 cm。R时钟跳变层的右腕最大D−R为26.42 cm，而连续层右肘D−R P95 0.198°，F−D为8.054°、F−R为8.053°。来源跳变和K在连续段增加的偏差同时存在，二者不能相互替代解释。',
 '本轮能定位失败发生在原生拟合、时序保持或候选选择；对固定模型不可达、范围／耦合约束和局部收敛的因果份额，仍需要独立证据。先保留test结果与这份诊断，任何后续改进应另开版本并登记已使用test的信息，不能继续称这批数据为未见留出。',
])]

md=['# 电视test：88路未通过的原因审查',f"\n证据快照：{data['created']}（Asia/Shanghai）。\n"]
page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>K4 电视测试：88路失败分析</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#203348;background:#f5f7fa;margin:0}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:30px}h2{margin-top:38px}p{max-width:1080px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border-bottom:1px solid #dbe3ea;padding:10px;text-align:left;vertical-align:top}th{background:#e7eef5}.scroll{overflow:auto}details{background:white;margin:12px 0;padding:12px;border:1px solid #dbe3ea}summary{cursor:pointer}small{color:#566879}a{color:#0969a4}.cards{display:flex;flex-wrap:wrap;gap:15px}.cards div{background:white;padding:18px;min-width:150px;border-top:4px solid #286a95}.cards b{display:block;font-size:30px}input{padding:10px;width:90%;margin:10px 0}pre{white-space:pre-wrap;word-break:break-all;font-size:12px}@media print{body{background:white}main{padding:0}details{break-inside:avoid}input{display:none}}</style><main><h1>K4 电视测试：88路失败分析</h1>',f'<small>快照 {html.escape(data["created"])} · 冻结方法，既有结果只读分析 · 可离线阅读</small><div class="cards"><div><b>4168 / 4256</b>实际选择结果通过</div><div><b>12 / 532</b>涉及的原始动作窗口</div><div><b>4</b>已确认选择器漏选</div><div><b>0</b>执行失败</div></div>']
for title,paras in sections:
    md+=['\n## '+title+'\n']+['\n'+p+'\n' for p in paras]
    page+=['<h2>'+html.escape(title)+'</h2>']+['<p>'+html.escape(p)+'</p>' for p in paras]
    tables=[]
    if title=='动作与格式分布':tables=[(['窗口','任务','原始动作','原时间窗 / s','未通过格式','部位与问题'],windowrows),(['格式','失败/总路数','失败率'],formatrows)]
    if title.startswith('拟合偏差'):tables=[(['部位','涉及失败路数（可重叠）'],bodyrows),(['样例','指标','D−R P95','F−D P95','F−R P95','F−R中位数'],paired)]
    for headers,rows in tables:md+=['\n'+table(headers,rows)];page+=[htmltable(headers,rows)]
md+=['\n## 可复核证据与复跑\n',f'- 完整只读证据：[evidence.json]({index["evidence"]})；包含完整／连续／来源跳变三层、D−R/F−D/F−R、原始／统一骨长、原生／导出、D／R时钟。\n',f'- 88路明细在 [HTML]({ROOT}/docs/20261121_k4_tv_failure_analysis.html)，可按窗口或格式过滤。\n',f'- 生成脚本：[analyze.py]({HERE}/analyze.py)、[report.py]({HERE}/report.py)；仅使用Python标准库读取既有JSON。未运行新的生物力学计算。\n',f'```bash\npython {HERE}/analyze.py --run {data["source"]} --output {index["evidence"]}\npython {HERE}/report.py\n```\n']
page+=['<h2>88路明细</h2><p>点击展开各路全部超标项。表内为F−R验收值；时序项显示D−R基线与允许增量。四套视图与时钟分别列出，不将重复指标相加计数。</p><input id="filter" placeholder="筛选：T075、Kimodo、肩、时序……"><div id="routes">']
for x in data['records']:
    label=f"{x['window']} · {names[x['method']]} · {x['failure_category']} · {x['selector_category']}"
    page+=['<details class="route"><summary>'+html.escape(label)+'</summary>', '<p>所选：'+html.escape(x['selected'])+'；最早失败层：'+x['first_failure_layer']+'；已访问通过候选：'+html.escape('、'.join(x['passing_visited']) or '0')+'；部位：'+html.escape('、'.join(x['regions']))+'</p>']
    metrics=[]
    for z in x['failed_metrics']:
        value=z.get('p95',z.get('candidate'));value=f'{value:.4f}' if isinstance(value,(float,int)) else str(value)
        limit=f"{z['baseline']:.4f} + {z['increase_limit']:g}" if 'increase_limit' in z else str(z.get('target','—'))
        metrics.append([z['view']+'/'+z['clock'],z['metric'],value,limit,f"{z.get('scored','—')}/{z.get('expected_observed','—')}"])
    page+=[htmltable(['视图/时钟','指标','F−R验收值','门槛/基线+增量','可评分/预期'],metrics),'<details><summary>既有证据路径</summary><pre>'+html.escape(json.dumps(x['provenance'],ensure_ascii=False,indent=2))+'</pre></details></details>']
page+=['</div><h2>证据范围</h2><p>详见仓库 develop_logs/notes/20261121_k4_tv_failure_analysis.md；大证据留在数据盘。页面内嵌88路明细，无外部脚本或网络依赖。精确高误差帧区间尚未重新计算；本报告按既有完整窗口统计定位问题。</p><script>document.getElementById("filter").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll(".route").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(q))});</script></main></html>']
(ROOT/'develop_logs/notes/20261121_k4_tv_failure_analysis.md').write_text('\n'.join(md))
(ROOT/'docs/20261121_k4_tv_failure_analysis.html').write_text('\n'.join(page))
print('Rendered 12 windows / 88 routes; notes and offline HTML.')
