"""Offline, synchronized common-coordinate R/D/F overlays; independent scoring only."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.motion import load,Motion
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points
from evalkit_k4_ablation_score import at,jump_mask

TEMPLATE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 开发窗口对照</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f2f5f8;color:#203245;margin:24px auto;max-width:1440px;padding:0 16px}h1{font-size:25px}select,button,input{font:inherit}button,select{padding:6px}nav{display:flex;gap:14px;flex-wrap:wrap;align-items:center;background:white;padding:12px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.tile{background:white;border:1px solid #cdd8e2;border-radius:8px;padding:8px}canvas{width:100%;height:360px;display:block}.legend span{margin-right:20px}details{margin-top:14px}pre{white-space:pre-wrap;font-size:12px}#seek{flex:1;min-width:200px}.muted{color:#53677c}table{border-collapse:collapse;width:100%;background:white;margin-top:15px}td,th{padding:6px;border:1px solid #d3dde4;text-align:left}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.grid{grid-template-columns:1fr}}@media print{nav{display:none}canvas{height:260px}body{margin:0}.grid{grid-template-columns:repeat(2,1fr)}}</style>
<h1>K4：原动作、退化输入与模型 FK 的同步对照</h1>
<p>用于检查独立运动学候选的局部残差与连续性。所有画面采用同一世界坐标、时间与米制尺度；显示公共评价节点，不做逐帧居中或原动作对齐补偿。旧候选仅作对照，当前候选尚待完整开发验收。</p>
<nav><label>窗口 <select id="case"></select></label><label>视角 <select id="view"><option value="front">世界 XY</option><option value="side">世界 ZY</option><option value="oblique" selected>斜视</option></select></label><button id="play">播放</button><input id="seek" type="range" min="0" step="1"><output id="time"></output></nav>
<p class="legend"><span style="color:#172b4d">● 原动作 R</span><span style="color:#168875">● 退化 D</span><span style="color:#bc4927">● 当前候选 F</span><span style="color:#7659ad">● 旧候选 FK</span><strong id="stratum"></strong></p>
<div class="grid"><div class="tile"><b>原动作 R</b><canvas></canvas></div><div class="tile"><b>退化输入 D</b><canvas></canvas></div><div class="tile"><b>当前候选 F</b><canvas></canvas></div><div class="tile"><b>端到端与旧候选叠加</b><canvas></canvas></div></div>
<p id="identity" class="muted"></p><div id="metrics"></div><details><summary>文件、坐标与核查依据</summary><pre id="evidence"></pre></details>
<script id="data" type="application/json">__DATA__</script><script>
const DATA=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('case'),seek=document.getElementById('seek'),view=document.getElementById('view'),play=document.getElementById('play'),can=[...document.querySelectorAll('canvas')];
DATA.cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.window+' / '+c.method;sel.append(o)});
const colors={R:'#172b4d',D:'#168875',F:'#bc4927',B:'#7659ad'};let playing=false,last=0;
function project(p){if(!p||p.some(x=>x===null))return null;let x=p[0],y=p[1],z=p[2];if(view.value==='front')return[x,y];if(view.value==='side')return[z,y];return[.707*x-.707*z,.92*y-.28*(x+z)];}
function draw(){const c=DATA.cases[+sel.value],i=+seek.value;document.getElementById('time').textContent=(c.times[i]-c.times[0]).toFixed(3)+' s · '+(i+1)+'/'+c.times.length;document.getElementById('stratum').textContent=c.tolerance[i]?'冻结来源跳变宽容段（保留显示）':'连续评分段';
const all=[];for(const poses of Object.values(c.positions))for(const frame of poses)for(const p of frame){let v=project(p);if(v)all.push(v)};const xs=all.map(x=>x[0]),ys=all.map(x=>x[1]);const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);let sets=[['R'],['D'],['F'],['B','R','F']];
can.forEach((canvas,k)=>{let w=canvas.clientWidth,h=canvas.clientHeight,ratio=devicePixelRatio||1;canvas.width=w*ratio;canvas.height=h*ratio;let g=canvas.getContext('2d');g.scale(ratio,ratio);g.fillStyle='white';g.fillRect(0,0,w,h);const scale=Math.min((w-35)/Math.max(xmax-xmin,.5),(h-35)/Math.max(ymax-ymin,.5));function xy(p){let v=project(p);return v?[(v[0]-(xmin+xmax)/2)*scale+w/2,h/2-(v[1]-(ymin+ymax)/2)*scale]:null}
g.strokeStyle='#e2e8ed';g.lineWidth=1;g.beginPath();g.moveTo(12,h-15);g.lineTo(12+.5*scale,h-15);g.stroke();g.fillStyle='#65768a';g.font='11px system-ui';g.fillText('0.5 m',12,h-20);
for(const key of sets[k]){const ps=c.positions[key][i];g.strokeStyle=colors[key];g.fillStyle=colors[key];g.lineWidth=key==='B'?1.5:2;g.setLineDash(key==='B'?[4,3]:[]);for(const [a,b] of DATA.edges){let p=xy(ps[a]),q=xy(ps[b]);if(p&&q){g.beginPath();g.moveTo(...p);g.lineTo(...q);g.stroke()}}g.setLineDash([]);for(const p of ps){let v=xy(p);if(v){g.beginPath();g.arc(...v,2.5,0,2*Math.PI);g.fill()}}}});}
function choose(){let c=DATA.cases[+sel.value];seek.max=c.times.length-1;seek.value=0;document.getElementById('identity').textContent=c.label;document.getElementById('evidence').textContent=JSON.stringify(c.evidence,null,2);let html='<table><tr><th>完整窗口连续层 P95</th><th>D−R</th><th>F−D</th><th>F−R</th></tr>';for(const row of c.metrics){html+='<tr><td>'+row.name+'</td>'+row.values.map(v=>'<td>'+(v===null?'无可评分观测':v.toFixed(3))+'</td>').join('')+'</tr>'}document.getElementById('metrics').innerHTML=html+'</table>';draw()}
sel.onchange=choose;seek.oninput=()=>{playing=false;play.textContent='播放';draw()};view.onchange=draw;window.onresize=draw;play.onclick=()=>{playing=!playing;play.textContent=playing?'暂停':'播放';last=0};function tick(now){if(playing){let c=DATA.cases[+sel.value],i=+seek.value,dt=1000*(c.times[Math.min(i+1,c.times.length-1)]-c.times[i]||c.times[1]-c.times[0]);if(!last||now-last>=dt){seek.value=(i+1)%c.times.length;last=now;draw()}}requestAnimationFrame(tick)}choose();requestAnimationFrame(tick);
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--baseline-condition',required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Keep visualization snapshots immutable')
    frozen=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    records={(x['window'],x['method']):x for x in json.loads((frozen/'artifact_verification.json').read_text())['records']}
    cases=[];edges=[]
    for row in json.loads((a.candidate/'panel.json').read_text())['records']:
        w,m=row['window'],row['method'];src=frozen/'inputs'/w/m/'candidate'
        d=load(src/'factorial/reference.npz' if m=='primal' else src,m,config=ROOT/'config/evalkit_k4_20261117/input.json')
        ref=frozen/'references'/f'{w}.npz'
        with np.load(ref) as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
        paths={'F':a.candidate/'views'/a.condition/'after'/w/m/'fk.npz',
               'B':a.baseline/'views'/a.baseline_condition/'after'/w/m/'fk.npz'}
        motions={'R':points(sample(r,d.times),True),'D':points(d)}
        for key,path in paths.items():
            with np.load(path) as z:f=Motion(z['positions'],list(z['names']),z['times'],'model_fk')
            motions[key]=points(at(f,d.times))
        names=list(motions['D']);arr={k:np.stack([v[n] for n in names],axis=1) for k,v in motions.items()}
        def finite(x):return np.where(np.isfinite(x),np.round(x,6),None).tolist()
        pairs=[('pelvis_mid','shoulder_mid'),('left_hip','right_hip'),('left_shoulder','right_shoulder'),('shoulder_mid','neck'),('neck','head')]
        for side in ('left','right'):
            for chain in [('hip','knee','ankle','foot'),('shoulder','elbow','wrist')]:
                pairs += [(side+'_'+x,side+'_'+y) for x,y in zip(chain[:-1],chain[1:])]
        edges=[(names.index(x),names.index(y)) for x,y in pairs]
        scorepath=a.candidate/'scores'/a.condition/'after'/f'{w}_{m}.json';scores=json.loads(scorepath.read_text())
        metrics=[]
        for title,path in [('骨盆线方向 / °',['directions_deg','pelvis_line']),('躯干方向 / °',['directions_deg','trunk_axis']),
                           ('左膝几何角 / °',['joint_angles_deg','left_knee']),('右膝几何角 / °',['joint_angles_deg','right_knee']),
                           ('右肩世界位置 / cm',['positions_cm','shoulders','right_shoulder','world']),
                           ('左足世界位置 / cm',['positions_cm','feet','left_foot','world'])]:
            values=[]
            for pair in ('D-R','F-D','F-R'):
                value=scores['comparisons'][pair]['continuous']
                for k in path:value=value[k]
                values.append(value['p95'])
            metrics.append({'name':title,'values':values})
        paths.update(R=ref,D=src)
        cases.append({'window':w,'method':m,'times':d.times.tolist(),'positions':{k:finite(x) for k,x in arr.items()},
            'tolerance':jump_mask(d.times,d,records[(w,m)]['source_jump_events']).tolist(),'metrics':metrics,
            'label':a.condition+'；旧候选 '+a.baseline_condition+'。角度目标5°，位置目标5cm；完整窗口与时序由独立结果验收。',
            'evidence':{'files':{k:str(v) for k,v in paths.items()},'score':str(scorepath),'coordinate':'right-handed world Y-up; metres; no view-specific registration',
                        'candidate_identity':json.loads((a.candidate/'identity.json').read_text()),'baseline_identity':json.loads((a.baseline/'identity.json').read_text())}})
    data=json.dumps({'cases':cases,'edges':edges},ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(TEMPLATE.replace('__DATA__',data))
    print(a.output,hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__=='__main__':main()
