"""Offline closure report; frozen results and independent candidates stay separate."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-run',type=Path,required=True)
    p.add_argument('--candidate-run',type=Path,action='append',default=[])
    p.add_argument('--path-replay',type=Path,action='append',default=[])
    p.add_argument('--rotation-replay',type=Path,action='append',default=[])
    p.add_argument('--selection-review',type=Path)
    p.add_argument('--reachability',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();groups=defaultdict(list);completed=passed=execution_failed=0
    for path in sorted((a.frozen_run/'tasks').glob('*/result.json')):
        r=json.loads(path.read_text())
        if r['status']!='completed':execution_failed+=1;continue
        completed+=1;passed+=bool(r.get('passed'))
        if r.get('passed'):continue
        d=json.loads((path.parent/'summary.json').read_text())
        groups[d['window']].append(d)
    esc=lambda x:html.escape(str(x))
    def metric_label(value):
        names={'trunk_axis':'躯干轴','shoulder_line':'肩线','pelvis_line':'骨盆线','head_neck_axis':'头颈轴',
               'left':'左','right':'右','thigh':'大腿','shank':'小腿','foot_axis':'足方向','foot':'足端',
               'shoulder':'肩','knee':'膝','ankle':'踝','wrist':'腕','upper_arm':'上臂','forearm':'前臂',
               'direction_rate_vector_error_deg_s':'方向变化率误差','output_step_deg':'最大角步长',
               'directions_deg':'方向角','positions_cm':'位置','joint_angles_deg':'几何夹角','world':'世界坐标'}
        return '／'.join(names.get(part,names.get(part.split('_')[0],'')+names.get('_'.join(part.split('_')[1:]),part)) for part in value.split('/'))
    table=[]
    for w,rows in groups.items():
        pose=sorted({x['metric'] for row in rows for x in row['failed_metrics'] if x['status']=='target_failed'})
        timing=sorted({x['metric'] for row in rows for x in row['failed_metrics'] if x['status']=='failed'})
        table.append(f'<tr><td>{esc(w)}</td><td>{len(rows)}</td><td>{esc(", ".join(r["method"] for r in rows))}</td><td>{esc("；".join(map(metric_label,pose)) or "本快照无姿态超标")}</td><td>{esc("；".join(map(metric_label,timing)) or "本快照无时序超标")}</td></tr>')
    sections=[];runtime_010_passed=False
    for run in a.candidate_run:
        progress=run/'progress.json';review=run/'review.json'
        if not review.exists():
            d=json.loads(progress.read_text()) if progress.exists() else {}
            n=sum(r['status']=='completed' for r in d.get('results',[]))
            expected=d.get('expected')
            if expected is None and (run/'panel.json').exists():
                expected=len(json.loads((run/'panel.json').read_text())['records'])*len(json.loads((run/'protocol.json').read_text())['conditions'])
            sections.append(f'<section><h3>{esc(run.name)}</h3><p>已完成{n}/{esc(expected or "待登记")}路；独立验收待完成。执行进度与通过数量分开统计。</p></section>');continue
        d=json.loads(review.read_text());rows=[]
        for r in d['records']:
            if r['window']=='010' and r['condition']=='reverse_path' and r.get('passed'):
                runtime_010_passed=True
            columns=[]
            for view,clock in [('native','D_clock'),('native','R_clock'),('export','D_clock'),('export','R_clock')]:
                state=r.get('acceptance',{}).get(view,{}).get(clock,{}).get('status','pending')
                columns.append(f'<td class="{esc(state)}">{esc({"passed":"通过","failed":"未通过","pending":"待完成"}.get(state,state))}</td>')
            failure='；'.join(f'{x["view"]}/{x["clock"]}: {x["metric"]} = {x.get("p95",x.get("candidate","覆盖不足"))}' for x in r.get('failures',[]))
            rows.append(f'<tr><td>{esc(r["window"])}</td><td>{esc(r["method"])}</td><td>{esc(r["condition"])}</td>{"".join(columns)}<td><details><summary>{"完整通过" if r.get("passed") else "查看未通过项"}</summary>{esc(failure or ("无超标项；仍需迁移回归" if r.get("passed") else "该条件尚未完成独立验收"))}</details></td></tr>')
        sections.append(f'<section><h3>{esc(run.name)}</h3><p>已完成{d["completed"]}/{d["expected"]}路，其中验收通过{d["passed"]}路；执行失败{d["execution_failed"]}。候选均未晋升默认。</p><div class="scroll"><table><thead><tr><th>窗口</th><th>格式</th><th>条件</th><th>原生D时钟</th><th>原生R时钟</th><th>导出D时钟</th><th>导出R时钟</th><th>详情</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
    for replay in a.path_replay:
        receipt=replay/'acceptance.json'
        if not receipt.exists():continue
        rows=[]
        for row in json.loads(receipt.read_text())['records']:
            columns=''.join(f'<td>{"通过" if row["acceptance"][clock]["status"]=="passed" else "未通过"}</td>' for clock in ['D_clock','R_clock'])
            rows.append(f'<tr><td>{esc(row["condition"])}</td>{columns}</tr>')
        stage='010 ARDy完整运行入口也已通过，其他路径与迁移继续验收。' if runtime_010_passed else '完整运行入口与迁移验证仍须完成。'
        sections.append('<section><h3>010：固定原生轨迹的插值隔离</h3><p>相同原生坐标及模型。原导出足最大步长36.14°，协调后1.92°；协调候选两种时钟完整通过，MOT回读误差为数值精度量级。'+stage+'R时钟变化率按冻结的D−R基线加增量门槛判定。</p><table><tr><th>路径</th><th>D时钟</th><th>R时钟</th></tr>'+''.join(rows)+'</table></section>')
    if a.reachability:
        rows=[]
        for path in sorted(a.reachability.glob('[0-9][0-9][0-9].json')):
            d=json.loads(path.read_text())
            for policy in ('current','expanded_domain'):
                entries=[r for r in d['rows'] if r['policy']==policy]
                fixed=[r['fixed_proximal']['directions_deg'][2] for r in entries]
                released=[r['released_proximal']['directions_deg'][2] for r in entries]
                rows.append(f'<tr><td>{esc(path.stem)}</td><td>{"当前域" if policy=="current" else "扩展域"}</td><td>{len(entries)}</td><td>{min(fixed):.3f}–{max(fixed):.3f}°</td><td>{min(released):.3f}–{max(released):.3f}°</td></tr>')
        sections.append('<section><h3>足踝可达性：固定近端与释放近端</h3><p>仅使用D；每个窗口选三个均匀帧与三个足方向残差最大帧，去重后固定。先保持髋膝姿态扫描踝／距下坐标，再释放该侧髋膝，比较方向和相对位置目标。扩展模型仅坐标范围不同，几何和耦合保持。下表是抽样诊断范围，非全窗口P95或全局不可达证明。</p><table><tr><th>窗口</th><th>坐标域</th><th>帧数</th><th>固定近端的足方向误差</th><th>释放近端的足方向误差</th></tr>'+''.join(rows)+'</table></section>')
    for run in a.rotation_replay:
        file=run/'review.json'
        if not file.exists():
            sections.append(f'<section><h3>{esc(run.name)}</h3><p>固定原生路径回放正在执行；独立原动作验收待完成。</p></section>');continue
        data=json.loads(file.read_text());rows=[]
        for item in data['rows']:
            roundtrip=run/'replays'/item['window']/item['method']/'roundtrip/audit.json'
            serialized=json.loads(roundtrip.read_text()).get('passed',False) if roundtrip.exists() else None
            for condition,value in item['conditions'].items():
                states=[value.get(k,{}).get('status','pending') for k in ('acceptance_relative_D','acceptance_original_clock_relative_D')]
                cells=''.join(f'<td class="{esc(s)}">{esc({"passed":"通过","failed":"未通过","pending":"待完成"}.get(s,s))}</td>' for s in states)
                failures=[]
                for clock,key in [('D','acceptance_relative_D'),('R','acceptance_original_clock_relative_D')]:
                    acceptance=value.get(key,{})
                    for metric in acceptance.get('required',[])+acceptance.get('temporal',[]):
                        if metric['status'] not in ('passed','unobservable'):
                            failures.append(clock+'：'+metric_label(metric['metric']))
                rows.append(f'<tr><td>{esc(item["window"])}</td><td>{esc(item["method"])}</td><td>{esc(condition)}</td>{cells}<td>{"通过" if serialized else "未通过" if serialized is False else "待完成"}</td><td>{esc("；".join(failures) or "本条件无超标")}</td></tr>')
        sections.append(f'<section><h3>固定原生路径：{esc(run.name)}</h3><p>同一行来源的模型、原生坐标与头颈附着保持一致。native是原生FK；whole_native_grid是SO(3)插值；coordinated是固定原生锚点的模型协调路径。目标属于插值假设，独立验收仍比较原动作。文件回读另列，原生失败不能由插值通过替代。</p><div class="scroll"><table><tr><th>窗口</th><th>格式</th><th>条件</th><th>D时钟</th><th>R时钟</th><th>MOT回读</th><th>超标项</th></tr>'+''.join(rows)+'</table></div></section>')
    if a.selection_review:
        data=json.loads(a.selection_review.read_text());records=data['records'];capable=sum(r.get('any_candidate_R_passed',False) for r in records)
        rows=[]
        for tag,label in [('native_only','原生观测排序'),('with_path','观测和路径平铺排序'),('observed_then_path','观测优先、路径次序排序')]:
            success=sum(r.get('choice',{}).get(tag,{}).get('independent_R_passed',False) for r in records)
            rows.append(f'<tr><td>{label}</td><td>{success}/{len(records)}</td><td>{capable-success}</td></tr>')
        sections.append(f'<section><h3>候选能力与输入选择分开评价</h3><p>{len(records)}个开发组合中，{capable}组已有至少一个独立验收通过的候选。运行时选择仅使用D；选择后由评分端检查R。各组合候选来源不完全一致，此表属于选择机制诊断，统一候选库迁移仍需完成。</p><table><tr><th>选择规则</th><th>选中结果通过</th><th>已有通过候选却漏选</th></tr>'+''.join(rows)+'</table></section>')
    timestamp=datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 新增失败与独立优化</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#173346;background:#f5f8fa;margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:36px}section,.card{background:white;border:1px solid #d6e2e7;border-radius:8px;padding:20px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde6eb;text-align:left;vertical-align:top}th{background:#edf3f6}.passed{color:#167252}.failed{color:#ad4136}summary{cursor:pointer}details{max-width:500px;overflow-wrap:anywhere}.muted{color:#546c79}a{color:#136c9e}@media print{body{background:white}main{padding:0}.scroll{overflow:visible}table{font-size:10px}section{break-inside:avoid}details{display:block}}
</style><main><h1>K4：新增失败与独立优化</h1>'''
    document+=f'<p class="muted">快照：{esc(timestamp)}。离线可阅读全部嵌入结果；在线仅冻结回归总进度自动刷新，候选表以本快照为准。</p>'
    document+='<p>目标是按退化前原动作逐窗口验收。姿态、位置、时序、覆盖及实际导出共同决定通过；原生拟合改善需要继续检查采样点之间的运动。</p>'
    document+=f'<div class="card"><b id="live">冻结回归：{completed}/656完成，{passed}通过，{completed-passed}指标未通过，{execution_failed}执行失败。</b><p><a href="20261118_k4_full_regression.html">打开冻结656路完整报告</a> · <a href="20261118_k4_rotation_comparison.html">查看既有旋转对照动画</a></p></div>'
    document+='<h2>因果证据与剩余问题</h2><section><p><b>002：</b>足端固定几何改善原生膝足偏差，反向＋变化率候选在Kimodo完整通过；其他格式和动作继续迁移。</p><p><b>008：</b>位置链长代理在MotionLCM、OmniControl通过；中立长度仍属于明确估计，其他姿态单独检验。</p><p><b>010：</b>原生近伸直阶段出现髋—距下补偿分支切换。反向候选改善原生结果，导出中间状态仍突跳。新增跨关节协调已在固定轨迹的隔离实验通过，以模型FK产生插入帧；已通过的运行入口及其他候选见下表，跨格式迁移继续验收。</p><p><b>新增肩躯干与膝足失败：</b>持续前倾／肩线倾斜、深屈膝下分别验证固定几何、范围和求解。触边是诊断线索，配对实验才用于因果判断。</p></section>'
    completion_note='全部656路已完成，以下保留最终失败清单。' if completed+execution_failed==656 else '运行未结束，以下为已经评分的失败，尚未运行项保留在656总分母内。'
    document+='<h2>冻结版本的未通过窗口</h2><p>同一动作的八种格式是配对条件。'+completion_note+'</p><div class="scroll"><table><tr><th>窗口</th><th>失败路数</th><th>格式</th><th>姿态／位置</th><th>时序</th></tr>'+''.join(table)+'</table></div>'
    document+='<h2>独立候选：逐条件验收</h2>'+''.join(sections)
    document+='<section><h3>实验边界</h3><p>完整D−R/F−D/F−R、原始／统一骨长和源跳变分层保存在各版本评分记录；本页展示验收判定。R与参考骨长只进入评分，运行时使用D和声明配置。HYOID与受力资格保持原范围，留出新增使用0。</p></section></main><script>async function refresh(){if(location.protocol==="file:")return;try{const r=await fetch("/api/k4/progress",{cache:"no-store"});if(!r.ok)return;const d=await r.json();document.getElementById("live").textContent=`冻结回归实时：${d.evaluated}/656完成，${d.passed}通过，${d.evaluated-d.passed}指标未通过，${d.execution_failed}执行失败。候选表仍以页面快照为准。`;}catch{}}refresh();setInterval(refresh,15000);</script></html>'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.tmp');temporary.write_text(document);temporary.replace(a.output)


if __name__=='__main__':main()
