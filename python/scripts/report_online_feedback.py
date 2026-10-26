"""Reconcile online traces with motions and update the existing survey in place."""
import json
from pathlib import Path
import numpy as np
from motiogen.feedback import score_root_goals

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'develop_logs/test_runs'
rows, motions, goals = [], {}, {}
for method, suffix, version in [('dart', '', 2), ('primal', '', 2), ('primal', '_h2', 2),
                                ('dart', '', 3), ('primal', '', 3), ('primal', '_h2', 3)]:
    paths = [RUNS / f'20261024_{method}_online_v{version}_{clock}_feedback{feedback}_seed{seed}{suffix}'
             for clock in ['paused', 'realtime'] for feedback in [0, 1] for seed in range(5)]
    if (suffix or version == 3) and not all((path / 'trace.json').exists() for path in paths):
        continue
    for path in paths:
        trace = json.loads((path / 'trace.json').read_text())
        xyz = np.load(path / 'motion.npz')['joints']
        assert xyz.shape == (300, 22, 3) and np.isfinite(xyz).all(), path
        task = score_root_goals(xyz, trace['goal_events'])
        for observed, recorded in zip(task, trace['task']):
            assert observed['success'] == recorded['success']
            assert abs(observed['final_error_m'] - recorded['final_error_m']) < 1e-5
        metrics = json.loads((path / 'metrics.json').read_text())['metrics']
        compute = [request['compute_seconds'] for request in trace['requests']]
        target_requests = [request for request in trace['requests']
                           if request['request_frame'] >= 150 and
                           np.allclose(request['goal'], trace['goal_events'][1]['position'])]
        initial_goal = np.asarray(trace['goal_events'][0]['position'])
        initial_root = initial_goal - [1, 0, 0]
        phase_errors = np.linalg.norm(xyz[150:, 0, :2] - np.asarray(trace['goal_events'][1]['position'])[:2], axis=1)
        row = {'method': method.upper() + (' horizon2' if suffix else '') + (' 接续调度' if version == 3 else ''),
               'clock': trace['mode'], 'feedback': int(trace['feedback']), 'seed': trace['seed'],
               'final_error_m': task[1]['final_error_m'], 'success': int(task[1]['success']),
               'end_hold': int((phase_errors[-15:] <= .3).all()),
               'recovery_seconds': task[1]['recovery_seconds'],
               'target_occupancy': float((phase_errors <= .3).mean()),
               'hold_fraction': trace['hold_fraction'], 'wall_seconds': trace['wall_seconds'],
               'startup_seconds': trace['startup_seconds'],
               'compute_mean_ms': float(np.mean(compute) * 1000),
               'compute_p95_ms': float(np.quantile(compute, .95) * 1000),
               'lookahead_seconds': trace['lookahead_seconds'],
               'plan_frames': trace['requests'][0]['plan_frames'],
               'goal_apply_delay_seconds': ((max(target_requests[0]['apply_frame'], target_requests[0]['request_frame']) - 150) / 30
                                            if target_requests else None),
               'gpu_allocated_gib': trace['gpu_peak_bytes'] / 2**30,
               'jerk_rms': metrics['jerk_rms']['value'],
               'trunk_inclination': metrics['trunk_inclination']['value'],
               'protocol_version': version, 'source': str(path.relative_to(ROOT))}
        rows.append(row)
        key = f"{row['method']}/{row['clock']}/{row['feedback']}/{row['seed']}"
        # Shared origin is fixed from task initial condition, not frame-by-frame centering.
        motions[key] = (xyz - initial_root).round(4).tolist()
        goals[key] = [(np.asarray(g['position']) - initial_root).round(4).tolist()
                      for g in trace['goal_events']]

assert len(rows) == 120, 'The final report requires all six complete 20-run cohorts'
metrics = ['final_error_m', 'success', 'end_hold', 'target_occupancy', 'hold_fraction',
           'compute_mean_ms', 'compute_p95_ms', 'jerk_rms', 'trunk_inclination', 'gpu_allocated_gib']
groups, effects = [], []
prefix_checks = []
rng = np.random.default_rng(20261024)
methods = sorted(set(row['method'] for row in rows))
for method in methods:
    for clock in ['paused', 'realtime']:
        for seed in range(5):
            paired_rows = [next(row for row in rows if (row['method'], row['clock'], row['feedback'], row['seed']) ==
                                (method, clock, flag, seed)) for flag in [0, 1]]
            a, b = [np.load(ROOT / row['source'] / 'motion.npz')['joints'] for row in paired_rows]
            difference = float(np.abs(a[:150] - b[:150]).max())
            prefix_checks.append({'method': method, 'clock': clock, 'seed': seed,
                                  'pre_event_max_difference_m': difference})
            if '接续调度' in method and clock == 'paused':
                assert difference < 1e-5, ('pre-event pairing mismatch', method, seed, difference)
        for flag in [0, 1]:
            selected = [row for row in rows if (row['method'], row['clock'], row['feedback']) == (method, clock, flag)]
            assert len(selected) == 5
            groups.append({'method': method, 'clock': clock, 'feedback': flag, 'n': 5,
                           **{key: float(np.mean([row[key] for row in selected])) for key in metrics}})
        paired = []
        for seed in range(5):
            pair = [next(row for row in rows if (row['method'], row['clock'], row['feedback'], row['seed']) ==
                         (method, clock, flag, seed)) for flag in [0, 1]]
            paired.append({key: pair[1][key] - pair[0][key] for key in metrics})
        for key in metrics:
            delta = np.array([row[key] for row in paired])
            interval = np.quantile(rng.choice(delta, (10000, 5)).mean(1), [.025, .975])
            effects.append({'method': method, 'clock': clock, 'metric': key,
                            'delta': float(delta.mean()), 'ci95': interval.tolist()})
summary = {'n_runs': len(rows), 'rows': rows, 'groups': groups, 'effects': effects,
           'prefix_checks': prefix_checks,
           'scope': 'one goal-switch scenario, five seeds; kinematic playback, no force solver',
           'success_definition': 'after switch, ever within XY radius 0.3m for 15 consecutive frames at 30fps',
           'end_hold_definition': 'last 15 frames remain within the same XY radius',
           'h_accounting': 'compute statistics include completed/applied requests; trailing unapplied request not included; not total GPU-hours',
           'pairing': 'seed, body, checkpoint and task matched within method; real-time request timestamps and subsequent RNG streams may differ'}
(RUNS / '20261024_online_feedback_summary.json').write_text(json.dumps(summary, indent=2))
template = '''<section id="online"><h2>目标切换：到达、保持和实时执行必须分开看</h2>
<p>__COUNT__ 次真实推理：每组 5 个 seed，分别比较“实时接收目标”与“只冻结目标输入”。两组都保留状态反馈通道；接续组使用已提交轨迹末端的预测姿态作为生成历史，实际观测另存，不把它说成未来实测。每段 10 秒、30 FPS，第 5 秒把目标从相对初始根节点 (+1, 0) m 切到 (−1, +0.5) m。</p>
<p>这是 B 方向的根节点导航任务，不是 A 方向的手部操作、物体接触或力学扰动实验。DART 使用预训练目标策略；PRIMAL 使用项目的目标到速度控制器。模型之间的差异还包含人体参数、策略与采样步数，不作跨模型因果排名。</p><p>“接续调度”是正式对照：暂停模式要求目标变化前两组轨迹逐帧一致。无后缀组和 horizon2 是早期调度诊断：自适应 lookahead 会改变重规划帧及噪声序列，部分配对在目标变化前已分叉，差值只能作为部署配置比较，不能归因于目标反馈单因素。所有结果保留，避免只展示修复后的有利样本。</p>
<div id="onlineFindings"></div>
<p>__HORIZON_FINDING__</p>
<label>方法 <select id="onlineMethod"></select></label> <label>时钟 <select id="onlineClock"><option value="paused">暂停计算</option><option value="realtime">真实时间</option></select></label>
<label>指标 <select id="onlineMetric"></select></label><div style="overflow:auto"><table id="onlineStats"></table></div>
<p id="onlineDelta"></p><p id="onlineLatency"></p><p>均值以全部 5 次为分母，包括任务失败。差值为目标反馈开启 − 冻结目标，区间为 seed 配对 bootstrap 95%；五种随机种子不代表任务多样性。成功指切换后曾连续 0.5 秒处于 0.3 m 目标区；最终保持单独要求最后 0.5 秒仍在区内。DART 原生成功奖励只要求生成窗口中曾进入目标区，并非驻停控制；这里的持续到达与保持要求更强，不冒充官方导航成绩。</p>
<h3>同步动作与路径检查</h3><p>左：冻结目标；右：接收新目标。共享坐标、时间和尺度；俯视橙色圆为当前目标 0.3 m 区域，正面圆仅标目标投影位置，不用于判定。浅蓝线为根节点完整路径。播放第 5 秒前后的转向与停帧，可检查数值结论。</p>
<label>Seed <select id="onlineSeed"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
<label>视角 <select id="onlineView"><option value="top">俯视 X-Y</option><option value="front">正面 X-Z</option></select></label>
<button id="onlinePlay">播放／暂停</button><input id="onlineFrame" type="range" min="0" max="299" value="150"><span id="onlineStamp"></span>
<div style="display:flex;flex-wrap:wrap"><canvas id="onlineLeft" width="510" height="430" style="width:50%;min-width:250px;background:#0b1220"></canvas><canvas id="onlineRight" width="510" height="430" style="width:50%;min-width:250px;background:#0b1220"></canvas></div>
<details><summary>公平性、调度及资源口径</summary><p>先预热，再预填充首段；首段生成耗时另记。实时模式按墙钟推进，迟到结果丢弃过期前缀，缓冲耗尽保持末帧；暂停模式在推理时停止环境时钟。lookahead 由首段推理耗时 ×1.25 +1帧自动确定，上限为预测窗口减1帧，不按任务成绩调参。</p><p>无后缀组从请求时的实际历史生成分支，再接入其尚未过期的部分；其与旧计划分叉会造成跳变。“接续调度”组从已提交队列末端的计划状态生成下一段，旧队列不被覆盖。计划状态不是未来实测：接口同时保留实际观测、观测帧与预测交接帧；目标仅使用请求时已经收到的信息。这是项目调度消融，不是模型权重变更或物理反馈。</p><p>PRIMAL 默认每次输出14帧；horizon2 每次输出28帧。后者改变计算成本与重规划频率，所以作为独立调度消融，不与默认组混合。短窗口重复初始化和落地投影亦属于项目外壳开销。DART 根据两帧交接历史恒速度外推补足原生前向差分特征；没有偷用未来真值。</p><p>RT 模式同 seed 仍可能因计算时序改变请求帧及后续噪声序列，这是部署级总效应，不是严格逐噪声锁定的纯算法差值。推理耗时包含完成并接收的请求；末尾未应用请求不计入此均值，不能据此宣称完整 GPU·h。显存是 allocator 峰值，不是设备总显存或 RAM。暂停/实时的 jerk 均按环境 30 FPS 计算，停帧与衔接跳变不会被删掉。</p></details>
<p><a href="../test_runs/20261024_online_feedback_summary.json">下载逐次结果、配对区间和证据路径</a> · 原生组件消融见上方 PRIMAL 报告。</p></section>
<script>
(()=>{const d=__DATA__,motion=__MOTION__,goals=__GOALS__,methods=__METHODS__,$=x=>document.getElementById(x);
const names={final_error_m:'切换后最终误差 (m)',success:'曾持续到达比例',end_hold:'最终保持比例',target_occupancy:'切换后在目标区的帧比例',hold_fraction:'缓冲耗尽停帧比例',compute_mean_ms:'单次推理均值 (ms)',compute_p95_ms:'单次推理 P95 的组均值 (ms)',jerk_rms:'Jerk RMS (m/s³)',trunk_inclination:'躯干倾角 (°)',gpu_allocated_gib:'GPU allocator 峰值 (GiB)'};
for(const m of methods)$('onlineMethod').add(new Option(m,m));if(methods.includes('DART 接续调度'))$('onlineMethod').value='DART 接续调度';for(const[k,v]of Object.entries(names))$('onlineMetric').add(new Option(v,k));
const group=(m,c,f)=>d.groups.find(x=>x.method===m&&x.clock===c&&x.feedback===f);
$('onlineFindings').innerHTML=methods.filter(m=>methods.some(v=>v.includes('接续调度'))?m.includes('接续调度'):true).map(m=>{const a=group(m,'realtime',0),b=group(m,'realtime',1),p=group(m,'paused',1);return `<p><strong>${m}</strong>：实时目标反馈的最终误差均值从 ${a.final_error_m.toFixed(2)} m 变为 ${b.final_error_m.toFixed(2)} m；持续到达 ${Math.round(b.success*5)}/5，最终保持 ${Math.round(b.end_hold*5)}/5。实时停帧 ${(b.hold_fraction*100).toFixed(1)}%；暂停计算时持续到达 ${Math.round(p.success*5)}/5。不能把曾经到达解读为稳定停驻。</p>`}).join('');
function stats(){const m=$('onlineMethod').value,c=$('onlineClock').value,k=$('onlineMetric').value;$('onlineStats').innerHTML='<tr><th>条件</th><th>'+names[k]+'</th><th>样本数</th></tr>'+[0,1].map(f=>`<tr><td>${f?'更新目标':'冻结目标'}</td><td>${group(m,c,f)[k].toFixed(4)}</td><td>5</td></tr>`).join('');const e=d.effects.find(x=>x.method===m&&x.clock===c&&x.metric===k);$('onlineDelta').textContent='配对差值 '+e.delta.toFixed(4)+'；95% 区间 ['+e.ci95.map(x=>x.toFixed(4)).join(', ')+']';const rows=d.rows.filter(r=>r.method===m&&r.clock===c&&r.feedback===1),lat=rows.map(r=>r.goal_apply_delay_seconds).filter(v=>v!==null),rec=rows.map(r=>r.recovery_seconds).filter(v=>v!==null),mean=a=>a.reduce((x,y)=>x+y,0)/a.length;$('onlineLatency').textContent='首次新目标动作执行延迟：'+(lat.length?mean(lat).toFixed(3)+' 环境秒':'无新目标动作')+'；到达起点时间（仅成功 '+rec.length+'/5）：'+(rec.length?mean(rec).toFixed(3)+' 环境秒':'5 秒观察窗内均未持续到达')+'。暂停模式的环境秒不包含等待计算。';draw();}
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
function draw(){const m=$('onlineMethod').value,c=$('onlineClock').value,s=$('onlineSeed').value,f=+$('onlineFrame').value,top=$('onlineView').value==='top';const keys=[0,1].map(flag=>`${m}/${c}/${flag}/${s}`);let minx=-1.5,maxx=1.5,miny=top?-1:0,maxy=2;for(const key of keys)for(const frame of motion[key])for(const p of frame){minx=Math.min(minx,p[0]);maxx=Math.max(maxx,p[0]);miny=Math.min(miny,p[top?1:2]);maxy=Math.max(maxy,p[top?1:2]);}const scale=Math.min(450/(maxx-minx+.5),350/(maxy-miny+.5));const project=p=>[255+(p[0]-(minx+maxx)/2)*scale,220-(p[top?1:2]-(miny+maxy)/2)*scale];keys.forEach((key,flag)=>{const ctx=$(flag?'onlineRight':'onlineLeft').getContext('2d');ctx.clearRect(0,0,510,430);ctx.strokeStyle='#31516a';ctx.beginPath();motion[key].forEach((frame,i)=>{const p=project(frame[0]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();const target=project(goals[key][f<150?0:1]);ctx.strokeStyle='#fb923c';ctx.beginPath();ctx.arc(...target,.3*scale,0,2*Math.PI);ctx.stroke();const pts=motion[key][f];ctx.strokeStyle='#7dd3fc';ctx.lineWidth=2;for(let j=1;j<22;j++){ctx.beginPath();ctx.moveTo(...project(pts[parents[j]]));ctx.lineTo(...project(pts[j]));ctx.stroke();}ctx.fillStyle='#fff';ctx.fillText(flag?'更新目标':'冻结目标',15,20);ctx.fillText('共享空间尺度 · m',15,410);});$('onlineStamp').textContent=(f/30).toFixed(2)+' s';}
let play=false;$('onlinePlay').onclick=()=>play=!play;setInterval(()=>{if(play){$('onlineFrame').value=(+$('onlineFrame').value+1)%300;draw()}},1000/30);for(const id of ['onlineMethod','onlineClock','onlineMetric'])$(id).onchange=stats;for(const id of ['onlineSeed','onlineView','onlineFrame'])$(id).oninput=draw;stats();})();</script>'''
short, long = [next(g for g in groups if g['method'] == method and g['clock'] == 'realtime' and g['feedback'] == 1)
               for method in ['PRIMAL 接续调度', 'PRIMAL horizon2 接续调度']]
delays = [np.mean([r['goal_apply_delay_seconds'] for r in rows if r['method'] == method
                  and r['clock'] == 'realtime' and r['feedback'] == 1])
          for method in ['PRIMAL 接续调度', 'PRIMAL horizon2 接续调度']]
horizon_finding = (f"PRIMAL 将窗口从 14 帧加到 28 帧后，目标反馈开启组的 jerk 均值为 "
                   f"{short['jerk_rms']:.2f} → {long['jerk_rms']:.2f} m/s³；首次新目标动作延迟为 "
                   f"{delays[0]:.2f} → {delays[1]:.2f} 秒，最终误差为 "
                   f"{short['final_error_m']:.2f} → {long['final_error_m']:.2f} m。"
                   "较长窗口有平滑收益，也有响应代价；不能只凭吞吐率选择窗口。")
section = template.replace('__HORIZON_FINDING__', horizon_finding).replace('__COUNT__', str(len(rows))).replace('__DATA__', json.dumps(summary)).replace('__MOTION__', json.dumps(motions, separators=(',', ':'))).replace('__GOALS__', json.dumps(goals)).replace('__METHODS__', json.dumps(methods))
destination = ROOT / 'develop_logs/notes/20261024_feedback_survey.html'
html = destination.read_text()
start, end = '<!-- ONLINE_RESULTS_START -->', '<!-- ONLINE_RESULTS_END -->'
prefix, rest = html.split(start)
_, suffix = rest.split(end)
destination.write_text(prefix + start + '\n' + section + '\n' + end + suffix)
print(json.dumps({'runs': len(rows), 'groups': groups}, indent=2))
