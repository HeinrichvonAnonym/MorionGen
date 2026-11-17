"""Offline R/D overlays of fixed development windows; no fitted motion shown."""
import argparse
import json
import gzip
import base64
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.degradation.metrics import points
from motiogen.degradation.reference import sample
from motiogen.evalkit.k.motion import Motion,load

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
keys=['000','012','017','020','044','047','070','074','077']
names=['pelvis_mid','shoulder_mid','neck','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist','left_hip','left_knee','left_ankle','left_foot','right_hip','right_knee','right_ankle','right_foot']
edges=[[0,1],[1,2],[2,3],[1,4],[4,5],[5,6],[1,7],[7,8],[8,9],[0,10],[10,11],[11,12],[12,13],[0,14],[14,15],[15,16],[16,17]]
data={}
for key in keys:
    score=json.loads((args.run/'scores'/f'{key}.json').read_text())
    with np.load(args.run/'references'/f'{key}.npz') as z:ref=Motion(z['positions'],list(z['names']),z['times'],'reference',z['rotations'],list(z['names']))
    data[key]={'recording':score['window']['recording_id'],'methods':{}}
    for method in score['methods']:
        folder=args.run/'inputs'/key/method/'candidate'
        motion=load(folder/'factorial/reference.npz' if method=='primal' else folder,method)
        a=points(sample(ref,motion.times),True);b=points(motion)
        output={'times':motion.times.tolist(),
             'r':np.round(np.stack([a[n] for n in names],1),6).tolist(),
             'd':np.round(np.stack([b[n] for n in names],1),6).tolist()}
        mask=(ref.times>=motion.times[0])&(ref.times<=motion.times[-1])
        raw=Motion(ref.positions[mask],ref.names,ref.times[mask],'reference')
        interp=sample(Motion(motion.positions,motion.names,motion.times,method),raw.times)
        a,b=points(raw,True),points(interp)
        original={'times':raw.times.tolist(),'r':np.round(np.stack([a[n] for n in names],1),6).tolist(),
                  'd':np.round(np.stack([b[n] for n in names],1),6).tolist()}
        data[key]['methods'][method]={'output':output,'original':original}
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>退化对齐核查：原动作与读回动作</title><style>
body{margin:24px auto;padding:0 18px;max-width:1180px;font:16px/1.65 system-ui;color:#172d42;background:#f7fafc}h1{font-size:26px}p{max-width:1000px}.panel{background:white;padding:16px;border:1px solid #d9e1e8;border-radius:10px}select,button,input{font:inherit;margin:6px;padding:5px}canvas{width:100%;height:520px;background:#fff;border:1px solid #d9e1e8}small{color:#526777}.blue{color:#216bcb}.orange{color:#dc7420}#record{overflow-wrap:anywhere}#scrub{width:65%}@media(max-width:600px){canvas{height:390px}body{margin:12px auto}}@media print{button,#scrub{display:none}}
</style><h1>退化对齐核查：原动作 R 与退化读回 D</h1>
<p>只展示退化转换，未运行 K 重构。两组骨架使用同一世界坐标、同一物理时刻；保留根运动。六种旋转格式采用明确声明、片段内固定的源体型骨架。</p>
<div class="panel"><label>开发窗口<select id="window"></select></label><label>格式<select id="method"></select></label><label>视角<select id="view"><option value="0">正向</option><option value="1">侧向</option><option value="2">俯视</option></select></label>
<label>时间轴<select id="clock"><option value="output">实际输出时刻</option><option value="original">原始时刻：D 插值重建</option></select></label>
<button id="play">暂停</button><label><input id="arrows" type="checkbox">误差箭头放大 20 倍</label>
<div id="record"></div><div><span class="blue">蓝：原动作</span>　<span class="orange">橙：退化读回</span>　<small>重合处会覆盖；箭头放大仅用于辨认误差，骨架位置始终真实。</small></div></div>
<canvas id="canvas" width="1100" height="520"></canvas><div><input id="scrub" type="range" min="0" max="100" value="0"><span id="time"></span></div><p id="error"></p>
<p><small>头颈线表示 neck→head 几何方向；手为腕点，足为踝／趾基点。两种位置格式缺少独立头部轴向旋转。这里的重合只验证原生动作表示的保持，不授予解剖精度或肌骨拟合资格。077 的压缩脊柱在保留节点之间重新分配弯曲，内部逐椎体姿态不在本图的验收范围。</small></p>
<p><small>原始时间轴保留采样间隔内的损失。原动作已有跳变的邻域在报告中单列宽容，画面仍完整显示。047 的首个间隔可用于检查此区别。</small></p>
<script>(async()=>{const bytes=Uint8Array.from(atob('__DATA__'),c=>c.charCodeAt(0));const data=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text()), edges=__EDGES__;
const w=document.getElementById('window'),m=document.getElementById('method'),v=document.getElementById('view'),c=document.getElementById('canvas'),ctx=c.getContext('2d'),s=document.getElementById('scrub');let playing=true,frame=0,last=0,seq,center,extent;
Object.keys(data).forEach(k=>w.add(new Option(k,k)));w.value='074';Object.keys(data[w.value].methods).forEach(k=>m.add(new Option(k,k)));m.value='kimodo';
function setup(){seq=data[w.value].methods[m.value][document.getElementById('clock').value];frame=0;s.max=seq.times.length-1;document.getElementById('record').textContent=data[w.value].recording;const all=seq.r.flat().concat(seq.d.flat());center=[0,1,2].map(i=>(Math.max(...all.map(p=>p[i]))+Math.min(...all.map(p=>p[i])))/2);extent=Math.max(...[0,1,2].map(i=>Math.max(...all.map(p=>p[i]))-Math.min(...all.map(p=>p[i]))));draw()}
function xy(p){let q=p.map((x,i)=>x-center[i]),a=v.value==='1'?2:0,b=v.value==='2'?2:1,scale=420/Math.max(extent,.1);return[550+q[a]*scale,260-q[b]*scale]}
function bone(points,color,dash){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);edges.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(...xy(points[a]));ctx.lineTo(...xy(points[b]));ctx.stroke()});points.forEach(p=>{ctx.beginPath();ctx.arc(...xy(p),3,0,Math.PI*2);ctx.fill()})}
function draw(){if(!seq)return;frame=Math.max(0,Math.min(seq.times.length-1,frame));ctx.clearRect(0,0,1100,520);bone(seq.r[frame],'#216bcb',[]);bone(seq.d[frame],'#dc7420',[5,3]);let errors=seq.r[frame].map((p,j)=>Math.hypot(...p.map((x,i)=>x-seq.d[frame][j][i]))*100);if(document.getElementById('arrows').checked){ctx.setLineDash([]);ctx.strokeStyle='#b12b6b';seq.r[frame].forEach((p,j)=>{ctx.beginPath();ctx.moveTo(...xy(p));ctx.lineTo(...xy(p.map((x,i)=>x+20*(seq.d[frame][j][i]-x))));ctx.stroke()})}s.value=frame;document.getElementById('time').textContent=seq.times[frame].toFixed(3)+' s';document.getElementById('error').textContent='当前帧：所示关节点最大位置误差 '+Math.max(...errors).toFixed(4)+' cm。';}
w.onchange=m.onchange=document.getElementById('clock').onchange=setup;v.onchange=draw;s.oninput=()=>{frame=+s.value;draw()};document.getElementById('arrows').onchange=draw;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};
function tick(t){if(playing&&seq&&t-last>1000*(seq.times[1]-seq.times[0])){frame=(frame+1)%seq.times.length;last=t;draw()}requestAnimationFrame(tick)}setup();requestAnimationFrame(tick);
})().catch(e=>document.getElementById('error').textContent='加载错误：'+e.message);</script></html>'''
args.output.parent.mkdir(parents=True,exist_ok=True)
payload=base64.b64encode(gzip.compress(json.dumps(data,separators=(',',':')).encode())).decode()
args.output.write_text(html.replace('__DATA__',payload).replace('__EDGES__',json.dumps(edges)))
