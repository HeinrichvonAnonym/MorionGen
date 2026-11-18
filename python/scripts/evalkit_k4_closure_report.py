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
    p.add_argument('--real-run',type=Path)
    p.add_argument('--real-geometry-run',type=Path)
    p.add_argument('--unified-run',type=Path)
    p.add_argument('--unified-live-port',type=int,default=8767)
    p.add_argument('--budget-review',type=Path)
    p.add_argument('--pool-visual',type=Path)
    p.add_argument('--evidence-report',type=Path)
    p.add_argument('--actual-ablation-run',type=Path,action='append',default=[])
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
    def fidelity_details(file):
        fidelity=json.loads(file.read_text());parts=[]
        def stat_cell(stat):
            value=stat.get('p95')
            return ('未知' if value is None else f'{value:.4f}')+f'（{stat.get("n",0)}/{stat.get("denominator",0)}）'
        for view,metrics in fidelity['views'].items():
            body=[]
            for kind in ('directions_deg','joint_angles_deg'):
                for name,stat in metrics[kind].items():
                    body.append(f'<tr><td>{esc(metric_label(name))}</td><td>{stat_cell(stat)}°</td><td>—</td></tr>')
            for name,values in metrics['positions_cm'].items():
                body.append(f'<tr><td>{esc(metric_label(name))}</td><td>{stat_cell(values["world"])}cm</td><td>{stat_cell(values["common_length_world"])}cm</td></tr>')
            for name,values in metrics['temporal'].items():
                body.append(f'<tr><td>{esc(metric_label(name))}变化率</td><td>{stat_cell(values["direction_rate_vector_error_deg_s"])}°/s</td><td>—</td></tr>')
            parts.append(f'<details><summary>{esc(view)}：F−D P95／有效帧数</summary><table><tr><th>指标</th><th>原始几何</th><th>统一D骨长</th></tr>'+''.join(body)+'</table></details>')
        return ''.join(parts)
    table=[]
    for w,rows in groups.items():
        pose=sorted({x['metric'] for row in rows for x in row['failed_metrics'] if x['status']=='target_failed'})
        timing=sorted({x['metric'] for row in rows for x in row['failed_metrics'] if x['status']=='failed'})
        table.append(f'<tr><td>{esc(w)}</td><td>{len(rows)}</td><td>{esc(", ".join(r["method"] for r in rows))}</td><td>{esc("；".join(map(metric_label,pose)) or "本快照无姿态超标")}</td><td>{esc("；".join(map(metric_label,timing)) or "本快照无时序超标")}</td></tr>')
    sections=[];runtime_010_passed=False
    for run in a.candidate_run:
        progress=run/'progress.json';review=run/'review.json'
        candidate_stages=[]
        for job in sorted((run/'tasks').glob('*/*/runtime_job.json')):
            payload=json.loads(job.read_text());fit=Path(payload['output'])
            if not (fit/'candidates').exists() or (job.parent/'status.json').exists():continue
            ready=sorted(path.parent.name for path in (fit/'candidates').glob('*/receipt.json'))
            candidate_stages.append(f'<li>{esc(job.parent.name)}：{len(ready)}个候选拟合已保存；整路选择及独立验收待完成。<details><summary>已保存候选</summary>{esc("、".join(ready) or "首个候选运行中")}</details></li>')
        stage_html='<ul>'+''.join(candidate_stages)+'</ul>' if candidate_stages else ''
        if not review.exists():
            d=json.loads(progress.read_text()) if progress.exists() else {}
            n=sum(r['status']=='completed' for r in d.get('results',[]))
            expected=d.get('expected')
            if expected is None and (run/'panel.json').exists():
                expected=len(json.loads((run/'panel.json').read_text())['records'])*len(json.loads((run/'protocol.json').read_text())['conditions'])
            sections.append(f'<section><h3>{esc(run.name)}</h3><p>已完成{n}/{esc(expected or "待登记")}路；独立验收待完成。执行进度与通过数量分开统计。</p>{stage_html}</section>');continue
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
        sections.append(f'<section><h3>{esc(run.name)}</h3><p>已完成{d["completed"]}/{d["expected"]}路，其中验收通过{d["passed"]}路；执行失败{d["execution_failed"]}。候选均未晋升默认。</p>{stage_html}<div class="scroll"><table><thead><tr><th>窗口</th><th>格式</th><th>条件</th><th>原生D时钟</th><th>原生R时钟</th><th>导出D时钟</th><th>导出R时钟</th><th>详情</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
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
    if a.real_run:
        rows=[]
        audit=json.loads((a.real_run/'input_audit.json').read_text())
        for item in audit['rows']:
            folder=a.real_run/item['method'];status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            state={'completed':'拟合与保真报告已生成','failed':'执行失败'}.get(status.get('status'),'运行中' if folder.exists() else '等待执行')
            if not status and (folder/'fit/candidates').exists():
                state+=f'：{len(list((folder/"fit/candidates").glob("*/receipt.json")))}个候选拟合已保存'
            source='原生旋转存在；共同段坐标声明待接通' if item['raw_segment_rotations'] and not item['declared_rig'] else '位置输入；独立段旋转不可用'
            detail='拟合完成后生成逐部位保真度'
            if (folder/'fidelity.json').exists():
                detail=fidelity_details(folder/'fidelity.json')
            rows.append(f'<tr><td>{esc(item["method"])}</td><td>{item["frames"]}</td><td>{esc(state)}</td><td>{esc(status.get("selected","待选择"))}</td><td>{esc(source)}</td><td>{detail}</td></tr>')
        sections.append('<section><h3>八份真实基线输出：单独检验F−D</h3><p>读取核查8/8成功。它们没有配对原动作R，不计入656路端到端结果；原生／实际导出保真度与文件回读分别记录。六旋转格式的数据实际存在，当前接口缺少旧样例的共同段坐标转换声明，不能把该工程缺口写成数据无旋转。详情中的分母保留完整输入时钟；未知项和拟合缺测继续保留。</p><div class="scroll"><table><tr><th>格式</th><th>输入帧</th><th>进度</th><th>选择</th><th>旋转接口状态</th><th>F−D详情</th></tr>'+''.join(rows)+'</table></div></section>')
    if a.real_geometry_run:
        protocol=json.loads((a.real_geometry_run/'protocol.json').read_text());rows=[]
        for entry in protocol['records']:
            for condition in protocol['conditions']:
                folder=a.real_geometry_run/condition/entry['method']
                state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                auditfile=folder/'actual_export/audit.json'
                audit=json.loads(auditfile.read_text()) if auditfile.exists() else {}
                fidelity=folder/'fidelity.json'
                detail=fidelity_details(fidelity) if fidelity.exists() else '拟合及评分待完成'
                rows.append(f'<tr><td>{esc(entry["method"])}</td><td>{esc(condition)}</td><td>{esc({"completed":"完成","failed":"执行失败"}.get(state.get("status"),"运行中" if folder.exists() else "等待执行"))}</td><td>{"通过" if audit.get("passed") else "待完成" if not audit else "未通过"}</td><td>{detail}</td></tr>')
        sections.append('<section><h3>真实位置输入：固定几何单因素对照</h3><p>control与total_proxy使用同一真实生成动作，仅改变有效链长估计的稳定性规则。前三级脊柱逐段检查，派生肩中点末段的变化传播到总链长；估计保持片段内固定，中立长度仍未知。两条件均保留，F−D保真度与受控原动作验收分开；此处没有配对R。</p><div class="scroll"><table><tr><th>格式</th><th>条件</th><th>进度</th><th>实际回读</th><th>原生／导出F−D</th></tr>'+''.join(rows)+'</table></div></section>')
    for run in a.actual_ablation_run:
        protocol=json.loads((run/'protocol.json').read_text());rows=[]
        for entry in protocol['records']:
            for condition in protocol['conditions']:
                folder=run/condition/entry['method']
                status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                state={'completed':'执行与评分完成','failed':'执行失败'}.get(status.get('status'),'运行中' if folder.exists() else '等待执行')
                fidelity=folder/'fidelity.json';detail=fidelity_details(fidelity) if fidelity.exists() else '完整窗口拟合与评分待完成'
                rows.append(f'<tr><td>{esc(entry["method"])}</td><td>{esc(condition)}</td><td>{esc(state)}</td><td>{detail}</td></tr>')
        sections.append(f'<section><h3>真实输出机制对照：{esc(run.name)}</h3><p>输入固定、条件预登记；已有候选与控制都保留。结果只有F−D，不进入656路的原动作通过率。几何框架或固定中心估计保留假设属性，不新增旋转观测资格。</p><details><summary>本组假设与控制变量</summary><pre>{esc(json.dumps({k:protocol.get(k) for k in ["hypothesis","factors","scoring"]},ensure_ascii=False,indent=2))}</pre></details><div class="scroll"><table><tr><th>格式</th><th>条件</th><th>进度</th><th>完整窗口F−D</th></tr>'+''.join(rows)+'</table></div></section>')
    timestamp=datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 新增失败与独立优化</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#173346;background:#f5f8fa;margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:36px}section,.card{background:white;border:1px solid #d6e2e7;border-radius:8px;padding:20px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dde6eb;text-align:left;vertical-align:top}th{background:#edf3f6}.passed{color:#167252}.failed{color:#ad4136}summary{cursor:pointer}details{max-width:500px;overflow-wrap:anywhere}.muted{color:#546c79}a{color:#136c9e}@media print{body{background:white}main{padding:0}.scroll{overflow:visible}table{font-size:10px}section{break-inside:avoid}details{display:block}}
</style><main><h1>K4：新增失败与独立优化</h1>'''
    document+=f'<p class="muted">快照：{esc(timestamp)}。离线可阅读全部嵌入结果；在线仅冻结回归总进度自动刷新，候选表以本快照为准。</p>'
    document+='<p>目标是按退化前原动作逐窗口验收。姿态、位置、时序、覆盖及实际导出共同决定通过；原生拟合改善需要继续检查采样点之间的运动。</p>'
    document+=f'<div class="card"><b id="live">冻结回归：{completed}/656完成，{passed}通过，{completed-passed}指标未通过，{execution_failed}执行失败。</b><p><a href="20261118_k4_full_regression.html">打开冻结656路完整报告</a> · <a href="20261118_k4_rotation_comparison.html">查看既有旋转对照动画</a> · <a href="20261118_k4_coordinated_path_comparison.html">查看新协调路径与原动作时钟动画</a></p></div>'
    if a.real_geometry_run and (a.real_geometry_run/'completion.json').exists():
        before=a.real_geometry_run/'control/motionlcm/fidelity.json';after=a.real_geometry_run/'total_proxy/motionlcm/fidelity.json'
        if before.exists() and after.exists():
            first=json.loads(before.read_text())['views']['native']['positions_cm'];second=json.loads(after.read_text())['views']['native']['positions_cm']
            document+=f'<section><h3>新增几何证据：真实MotionLCM的固定模型匹配</h3><p>仅改变有效链长估计的准入规则，右肩F−D位置P95从{first["right_shoulder"]["world"]["p95"]:.3f}降到{second["right_shoulder"]["world"]["p95"]:.3f}cm；头／颈统一D骨长位置从{first["head"]["common_length_world"]["p95"]:.3f}降到{second["head"]["common_length_world"]["p95"]:.3f}cm。模型几何仍为片段内固定。此例支持派生肩中点的波动触发旧几何回退、进而影响拟合的解释。</p><p>这是单份真实输出的输入保真度对照，没有配对原动作R；两条件完整指标见后文。中立长度仍属未知，当前冻结候选库保持原值，开发集迁移及采样间路径另验。</p></section>'
    if a.unified_run and (a.unified_run/'progress.json').exists():
        progress=json.loads((a.unified_run/'progress.json').read_text())
        active_protocol=json.loads((a.unified_run/'protocol.json').read_text())
        active_title=esc(active_protocol.get('report_title','统一候选：完整656路'))
        active_page=esc(Path(active_protocol['html_report']).name)
        states={'waiting_for_registered_runs':'等待前序实验及独立验收','running':'运行中','validation_stage_failed':'阶段未通过，后续待执行','completed':'本版本执行结束'}
        document+=f'<section><h3>{active_title}</h3><p>{esc(states.get(progress.get("state"),progress.get("state")))}；已完成{progress.get("completed",0)}/656，通过{progress.get("passed",0)}/656，执行失败{progress.get("failed",0)}。这是同一新实现的完整开发回归，按27个统一检查、38个格式缺口、其余591路依次验收；每阶段保留全部分母。</p><p><a href="http://127.0.0.1:{a.unified_live_port}/docs/{active_page}">打开当前版本动态报告</a> · <a href="{active_page}">离线快照</a>。本卡片是本页生成时的快照；实时进度见动态报告。旧656结果继续单独保留。</p></section>'
    if a.evidence_report and a.evidence_report.exists():
        document+=f'<p><a href="{esc(a.evidence_report.name)}">打开当前版本逐路完整指标快照</a>：原生与实际导出、D／R时钟、三类误差及骨长口径可分别查看；保留656路完整分母。</p>'
    if a.budget_review:
        budget=json.loads(a.budget_review.read_text())['summary']
        document+=f'<section><h3>候选计算预算：单独验证输入选择</h3><p>按预登记顺序，首个覆盖完整、所有输入及导出路径指标均低于原目标0.8倍的候选可以提前结束搜索；未满足时仍执行原完整池。只读回放{budget["records"]}条原生／组合配对记录，覆盖{budget["unique_window_format_pairs"]}个窗口—格式组合、{budget["unique_original_windows"]}个原始动作；独立验收{budget["passed"]}通过、{budget["failed"]}未通过、{budget["unknown"]}待验。候选数量节省尚不能换算实际运行时间。该功能在默认接口中关闭，在独立预算v3中显式启用；当前完整开发回归会验证实际执行和656路结果。旧完整池与全部已生成候选保留。</p></section>'
    if a.pool_visual and a.pool_visual.exists():
        document+=f'<p><a href="{esc(a.pool_visual.name)}">查看当前实际选择：原动作、退化输入、模型FK与旧冻结结果四组动画</a>。使用原动作时钟及共同世界坐标；各窗口验收和待完成数量在动画页中分别标明。</p>'
    document+='<h2>因果证据与剩余问题</h2><section><p><b>002：</b>六旋转格式已在反向＋变化率候选通过局部验收。两位置格式采用输入估计的片段内固定足端，再配合已有链长代理，通过完整原生／导出及两套时钟；足端单因素保留为肩线边界失败对照。统一候选选择与广泛迁移继续检验。</p><p><b>008：</b>位置链长代理在MotionLCM、OmniControl通过；中立长度仍属于明确估计，其他姿态单独检验。</p><p><b>010：</b>原生近伸直阶段出现髋—距下补偿分支切换。反向候选改善原生结果，导出中间状态仍突跳。新增跨关节协调已在固定轨迹的隔离实验通过，以模型FK产生插入帧；已通过的运行入口及其他候选见下表，跨格式迁移继续验收。</p><p><b>新增肩躯干与膝足失败：</b>持续前倾／肩线倾斜、深屈膝下分别验证固定几何、范围和求解。触边是诊断线索，配对实验才用于因果判断。</p></section>'
    if a.rotation_replay:
        document+='<section><h3>本轮路径与支持标记核查</h3><p><b>032：</b>保持原生拟合不变，全身协调路径在Kimodo、MotionLCM通过两套时钟及文件回读；原SO(3)导出在R时钟仍有时序超标。</p><p><b>010 CHOIS：</b>进一步查到节点别名错误：先验手端点与真实wrist_body腕点在统一命名后发生支持标记混淆，旧路径求解因此整段跳过。此前“缺腕观测”的阶段解释已修正。新版本让位置与支持元数据共用别名映射；原始输入与评分口径保持。此前部分目标及腿部限定候选完整保留，新的完整支持回放见下表。</p><p><b>073、077：</b>073通过链长几何、整段目标及肩轴向域的配对对照获得完整通过候选；077进一步通过保持FK的等价欧拉坐标切换解除局部求解障碍。局部验收与数值收敛分别报告，优化预算耗尽仍保留；默认配置尚未晋升，23组合统一候选库正在检验运行时能否选对结果。</p></section>'
    completion_note='全部656路已完成，以下保留最终失败清单。' if completed+execution_failed==656 else '运行未结束，以下为已经评分的失败，尚未运行项保留在656总分母内。'
    document+='<h2>冻结版本的未通过窗口</h2><p>同一动作的八种格式是配对条件。'+completion_note+'</p><div class="scroll"><table><tr><th>窗口</th><th>失败路数</th><th>格式</th><th>姿态／位置</th><th>时序</th></tr>'+''.join(table)+'</table></div>'
    document+='<h2>独立候选：逐条件验收</h2>'+''.join(sections)
    document+='<section><h3>实验边界</h3><p>完整D−R/F−D/F−R、原始／统一骨长和源跳变分层保存在各版本评分记录；本页展示验收判定。R与参考骨长只进入评分，运行时使用D和声明配置。HYOID与受力资格保持原范围，留出新增使用0。</p></section></main><script>async function refresh(){if(location.protocol==="file:"||location.port!=="8766")return;try{const r=await fetch("/api/k4/progress",{cache:"no-store"});if(!r.ok)return;const d=await r.json();document.getElementById("live").textContent=`冻结回归实时：${d.evaluated}/656完成，${d.passed}通过，${d.evaluated-d.passed}指标未通过，${d.execution_failed}执行失败。候选表仍以页面快照为准。`;}catch{}}refresh();setInterval(refresh,15000);</script></html>'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.tmp');temporary.write_text(document);temporary.replace(a.output)


if __name__=='__main__':main()
