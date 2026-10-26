"""Build an inspectable paired ablation report from completed PRIMAL runs."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rows = []
motions = {}
for seed in range(5):
    directory = ROOT / f'develop_logs/test_runs/20261024_primal_feedback_f300_seed{seed}'
    for timing in json.loads((directory / 'timing.json').read_text()):
        variant = timing['variant']
        metrics = json.loads((directory / f'{variant}_metrics.json').read_text())['metrics']
        xyz = np.load(directory / f'{variant}_canonical.npz')['joints']
        feet = xyz[:, [10, 11]]
        speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * 30
        contact = feet[:-1, :, 2] < .05
        row = dict(timing)
        row.update({k: v['value'] for k, v in metrics.items() if v['status'] == 'measured'})
        row['rtf'] = timing['seconds'] / (len(xyz) / 30)
        row['height_based_foot_speed'] = float(speed[contact].mean()) if contact.any() else None
        rows.append(row)
        motions[f'{seed}/{variant}'] = xyz.round(4).tolist()
variants = [f'reproject{r}_inertialize{i}' for r, i in [(0, 0), (1, 0), (0, 1), (1, 1)]]
keys = ['jerk_rms', 'trunk_inclination', 'left_upper_arm_elevation', 'right_upper_arm_elevation', 'height_based_foot_speed', 'rtf']
means = {v: {k: float(np.mean([r[k] for r in rows if r['variant'] == v])) for k in keys} for v in variants}
effects = {}
rng = np.random.default_rng(20261024)
for v in variants[1:]:
    effects[v] = {}
    for key in keys:
        delta = np.array([next(r[key] for r in rows if r['seed'] == seed and r['variant'] == v)
                          - next(r[key] for r in rows if r['seed'] == seed and r['variant'] == variants[0]) for seed in range(5)])
        ci = np.quantile(rng.choice(delta, (10000, 5), replace=True).mean(axis=1), [.025, .975])
        effects[v][key] = {'paired_mean_difference': float(delta.mean()), 'bootstrap_95_ci': ci.tolist()}
summary = {'n_seeds': 5, 'n_runs': len(rows), 'means': means, 'effects_vs_no_projection_no_inertialization': effects,
           'scope': 'one adapted standing initial condition; five random seeds; 294 frames/run; no online goal changes',
           'foot_definition': 'mean foot speed when previous frame foot height < 5cm; geometric proxy, not force-based contact'}
(ROOT / 'develop_logs/test_runs/20261024_primal_feedback_summary.json').write_text(json.dumps(summary, indent=2))
template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PRIMAL 配对反馈消融</title>
<style>body{font:16px/1.6 system-ui;background:#111827;color:#e5edf5;max-width:1200px;margin:auto;padding:25px}section{background:#1e293b;padding:20px;border-radius:12px;margin:20px 0}a{color:#7dd3fc}td,th{padding:10px;border-bottom:1px solid #475569;text-align:left}select,button,input{padding:8px;margin:5px}canvas{background:#0b1220;width:48%;border-radius:10px}table{width:100%}.scroll{overflow:auto}</style>
<h1>PRIMAL：重投影与惯性平滑的配对消融</h1><p>2026-10-24 · 20 次真实 GPU 推理 · 5 个配对 seed</p>
<section><h2>实验回答什么</h2><p>同一公开 checkpoint、EMA、初态、50 步采样和随机 seed，分别开关关节重投影与惯性平滑；落地投影始终开启。测试它们对运动平滑、姿态及生成速度的收益与代价。</p><p>原生入口通过上一段生成状态反馈，不调用外部物理引擎。本实验初态来自 DART stand.pkl，旋转至 Y-up，使用中性零形状；每次生成 294 帧。结论适用于这一初态下的五个随机种子，不能代替多任务、目标突变或外力恢复实验。</p></section>
<section><h2>本组实验的结论</h2><p>惯性平滑降低了运动急促程度：单独开启时，平均 jerk 从 40.23 降到 18.15 m/s³，配对下降区间不跨零。单独重投影反而把 jerk 提高到 46.47；两者同时开启降到 15.66。结果说明重投影与平滑存在交互，不能把“纠正更多”直接等同于“动作更好”。</p><p>两者同时开启还把低足高帧足速从 0.143 降到 0.075 m/s，代价是 RTF 从 0.438 增至 0.451。躯干和抬臂没有一致改善，因此本组支持平滑性收益，不支持人体工学全面改善。</p></section>
<section><h2>均值与配对差值</h2><label>指标 <select id="metric"></select></label><div class="scroll"><table id="stats"></table></div><p>差值基准为两开关均关闭，95% 区间通过 seed 层配对 bootstrap 得到。五个 seed 对尾部风险的估计有限。姿态与足部指标是几何代理，不是关节负载。</p><p>全部变体先经过统一预热；RTF 包含原生生成入口开销，不包含模型加载与随后另算的统一指标。显存记录为 PyTorch allocator 峰值。</p></section>
<section><h2>同步动作检查</h2><p>左侧固定为两开关均关闭；右侧选择消融变体。共享帧、视角与空间尺度；坐标为米、右手 Z-up。</p><label>Seed <select id="seed"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label><label>右侧 <select id="variant"></select></label><label>视角 <select id="view"><option value="0">正面 X-Z</option><option value="1">侧面 Y-Z</option></select></label><button id="play">播放／暂停</button><input id="frame" type="range" min="0" max="293" value="0"><span id="stamp"></span><div><canvas id="left" width="550" height="500"></canvas><canvas id="right" width="550" height="500"></canvas></div></section>
<section><h2>证据</h2><p><a href="../test_runs/20261024_primal_feedback_summary.json">结构化汇总与置信区间</a> · <a href="https://github.com/yz-cnsdqz/primal-release">官方实现</a> · <a href="20261024_feedback_survey.html">总体调研报告</a></p><p>逐 case 输入来源、耗时、动作文件和指标位于 develop_logs/test_runs/20261024_primal_feedback_f300_seed0 至 seed4。</p></section>
<script>const data=__DATA__,motions=__MOTIONS__,variants=__VARIANTS__;const labels={jerk_rms:'Jerk RMS (m/s³)',trunk_inclination:'躯干倾角 (°)',left_upper_arm_elevation:'左上臂抬升 (°)',right_upper_arm_elevation:'右上臂抬升 (°)',height_based_foot_speed:'低足高帧足速 (m/s)',rtf:'RTF'};
const $=id=>document.getElementById(id);for(const[k,v]of Object.entries(labels))$('metric').add(new Option(v,k));for(const v of variants)$('variant').add(new Option(v,v));$('variant').value=variants[3];function stats(){const k=$('metric').value;$('stats').innerHTML='<tr><th>变体</th><th>均值</th><th>相对基准差值 [95% CI]</th></tr>'+variants.map((v,i)=>{const e=data.effects_vs_no_projection_no_inertialization[v]?.[k];return `<tr><td>${v}</td><td>${data.means[v][k].toFixed(4)}</td><td>${e?e.paired_mean_difference.toFixed(4)+' ['+e.bootstrap_95_ci.map(x=>x.toFixed(4)).join(', ')+']':'基准'}</td></tr>`}).join('')}
const parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];function draw(id,v){const ctx=$(id).getContext('2d'),pts=motions[$('seed').value+'/'+v][+$('frame').value],axis=+$('view').value;ctx.clearRect(0,0,550,500);ctx.strokeStyle='#475569';ctx.beginPath();ctx.moveTo(0,440);ctx.lineTo(550,440);ctx.stroke();const p=q=>[275+q[axis]*160,440-q[2]*160];ctx.strokeStyle='#7dd3fc';ctx.lineWidth=3;for(let j=1;j<22;j++){const a=p(pts[parents[j]]),b=p(pts[j]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.fillStyle='#e5edf5';ctx.fillText(v,10,20)}function render(){draw('left',variants[0]);draw('right',$('variant').value);$('stamp').textContent=(+$('frame').value/30).toFixed(2)+' s'}let playing=false;$('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){$('frame').value=(+$('frame').value+1)%294;render()}},1000/30);for(const id of ['seed','variant','view','frame'])$(id).oninput=render;$('metric').onchange=stats;stats();render();</script></html>'''
page = template.replace('__DATA__', json.dumps(summary)).replace('__MOTIONS__', json.dumps(motions, separators=(',', ':'))).replace('__VARIANTS__', json.dumps(variants))
destination = ROOT / 'develop_logs/notes/20261024_primal_feedback_ablation.html'
destination.write_text(page)
print(json.dumps(summary, indent=2))
print(destination)
