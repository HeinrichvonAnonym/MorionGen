"""Offline selected-pool R/D/F inspection at the original reference clock.

This reads existing FK and independent scores; it does not fit or select methods.
"""
import argparse,hashlib,json,sys,gzip,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from evalkit_k4_window_visuals import TEMPLATE
import numpy as np
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.metrics import points
from evalkit_k4_ablation_score import at,jump_mask


def read(path):
    with np.load(path) as z:return Motion(z['positions'],list(z['names']),z['times'],'fk')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('run','baseline','records','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Retain previous visual snapshots; use a new output')
    registry=json.loads(a.records.read_text())
    frozen=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    artifacts={(x['window'],x['method']):x for x in json.loads((frozen/'artifact_verification.json').read_text())['records']}
    review={(x['window'],x['method']):x for x in json.loads((a.run/'review.json').read_text())['records'] if x['condition']=='pool'}
    specs=[('骨盆线方向 / °',('directions_deg','pelvis_line')),('躯干方向 / °',('directions_deg','trunk_axis')),
           ('肩线方向 / °',('directions_deg','shoulder_line')),('头颈方向 / °',('directions_deg','head_neck_axis'))]
    for side,title in [('left','左'),('right','右')]:
        for part,part_title in [('upper_arm','上臂'),('forearm','前臂'),('thigh','大腿'),('shank','小腿'),('foot_axis','足')]:
            specs.append((title+part_title+'方向 / °',('directions_deg',side+'_'+part)))
        for joint,title2 in [('knee','膝'),('elbow','肘')]:specs.append((title+title2+'几何角 / °',('joint_angles_deg',side+'_'+joint)))
        for region,node,title2 in [('shoulders','shoulder','肩'),('hands','wrist','腕'),('knees','knee','膝'),('feet','ankle','踝'),('feet','foot','足')]:
            specs.append((title+title2+'世界位置 / cm',('positions_cm',region,side+'_'+node,'world')))
    for node,title in [('neck','颈'),('head','头')]:specs.append((title+'世界位置 / cm',('positions_cm','head_neck',node,'world')))
    cases=[];missing=[];hashes={str(a.records):hashlib.sha256(a.records.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for row in registry['records']:
        w,m=row['window'],row['method'];key=w+'_'+m
        if review.get((w,m),{}).get('status')!='completed':missing.append(key);continue
        fit=a.run/'fits/pool'/w/m;selection=json.loads((fit/'candidate_selection.json').read_text())
        old=json.loads((a.baseline/'tasks'/key/'summary.json').read_text());oldfit=Path(old['candidate_selection_file']).parent
        src=frozen/'inputs'/w/m/'candidate';ref=frozen/'references'/f'{w}.npz'
        d=load(src/'factorial/reference.npz' if m=='primal' else src,m,config=ROOT/'config/evalkit_k4_20261117/input.json')
        r=read(ref);clock=r.times
        motions={'R':points(r,True),'D':points(at(d,clock)),'F':points(at(read(fit/'fit.npz'),clock)),'B':points(at(read(oldfit/'fit.npz'),clock))}
        names=list(motions['D']);positions={k:np.stack([v[n] for n in names],1) for k,v in motions.items()}
        scorepath=a.run/'scores/pool/export'/(key+'.json');score=json.loads(scorepath.read_text())['original_clock']
        if len(clock)!=score['clock']['frames']:raise ValueError('Visual and original score clocks differ')
        np.testing.assert_allclose([clock[0],clock[-1]],[score['clock']['start_s'],score['clock']['end_s']],rtol=0,atol=1e-9)
        metrics=[]
        for title,path in specs:
            values=[]
            for pair in ('D-R','F-D','F-R'):
                value=score['comparisons'][pair]['continuous']
                for part in path:value=value[part]
                values.append(value['p95'])
            metrics.append({'name':title,'values':values})
        metrics.append({'name':'头相对胸完整旋转 / °（需独立旋转观测）','values':[score['head_neck_rotation_comparisons'][pair]['continuous']['p95'] for pair in ('D-R','F-D','F-R')]})
        files={'R':ref,'D':src,'F':fit/'fit.npz','B':oldfit/'fit.npz','scores':scorepath,'selection':fit/'candidate_selection.json'}
        for file in files.values():
            if file.is_file():hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        rr=review[w,m]
        cases.append(dict(window=w,method=m,times=clock.tolist(),positions={k:np.where(np.isfinite(v),np.round(v,6),None).tolist() for k,v in positions.items()},
            tolerance=jump_mask(clock,d,artifacts[w,m]['source_jump_events']).tolist(),metrics=metrics,
            label=f"实际选择：{selection['selected']}；旧冻结：{old['selected']}；独立完整验收：{'通过' if rr['passed'] else '未通过'}。",
            evidence={'files':{k:str(v) for k,v in files.items()},'acceptance':rr['acceptance'],'export_audit':rr['export_audit'],
                      'coordinates':'World Y-up metres; reference clock; no root/scale registration for display; D and exported F positions linearly sampled exactly as independent scoring. This sampling is a display/score operation, not a new K fit.',
                      'position_rounding_m':1e-6,'score_precision':'Original stored scores, unrounded'}))
    if not cases:raise ValueError('No independently reviewed completed cases')
    pairs=[('pelvis_mid','shoulder_mid'),('left_hip','right_hip'),('left_shoulder','right_shoulder'),('shoulder_mid','neck'),('neck','head')]
    for side in ('left','right'):
        for chain in [('hip','knee','ankle','foot'),('shoulder','elbow','wrist')]:pairs.extend((side+'_'+x,side+'_'+y) for x,y in zip(chain[:-1],chain[1:]))
    data={'cases':cases,'edges':[(names.index(x),names.index(y)) for x,y in pairs],'manifest':registry,'pending':missing,'source_sha256':hashes}
    template=TEMPLATE.replace('旧候选仅作对照，当前候选尚待完整开发验收。',f'当前显示预登记{len(registry["records"])}组中的{len(cases)}组，待完成：{", ".join(missing) or "无"}。完整656路仍单独验收，页面不代表全量通过。')
    template=template.replace('完整窗口连续层 P95','原动作时钟 · 连续层 P95（完整窗口及覆盖见详情）')
    template=template.replace('<script id="data" type="application/json">','<script id="data" type="application/octet-stream">')
    template=template.replace("const DATA=JSON.parse(document.getElementById('data').textContent),", "(async()=>{const packed=Uint8Array.from(atob(document.getElementById('data').textContent),c=>c.charCodeAt(0));const DATA=JSON.parse(await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).text()),")
    template=template.replace('choose();requestAnimationFrame(tick);\n</script>', 'choose();requestAnimationFrame(tick);})();\n</script>')
    payload=base64.b64encode(gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(template.replace('__DATA__',payload))
    print(json.dumps({'output':str(a.output),'shown':len(cases),'registered':len(registry['records']),'pending':missing}))

if __name__=='__main__':main()
