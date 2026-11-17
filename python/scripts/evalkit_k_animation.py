"""Build an offline synchronized R/D/F viewer from frozen evaluation arrays."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import NAMES,PARENTS,normalize
from motiogen.evalkit.k.scoring import angles,region
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
SUMMARY=json.loads((ROOT/'config/evalkit_k33_20261117/summary.json').read_text())
OUT=ROOT/'docs/20261117_k_motion_comparison.html'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def case(path,method,window,version,title,focus,description,key='F:new'):
    with np.load(path) as z:
        t=z['times'];data={k:z[k].copy() for k in ['R','D']};data['F']=z[key].copy();lens=z['reference_lengths']
    assert np.all(np.diff(t)>0) and all(x.shape==(len(t),16,3) and np.isfinite(x).all() for x in data.values())
    ar={k:angles(v) for k,v in data.items()};metrics=[k for k in ar['R'] if region(k)==focus]
    stages={label:float(np.mean([np.percentile(abs(ar[a][m]-ar[b][m]),95) for m in metrics])) for label,a,b in [('degradation','D','R'),('fit','F','D'),('end','F','R')]}
    # Every scored frame retained. Six decimal places bound coordinate error to 0.5 micrometre.
    packed={mode:{k:np.round(v if mode=='raw' else normalize(v,lens),6).tolist() for k,v in data.items()} for mode in ['raw','normalized']}
    origin=np.median(data['R'][:,0],axis=0)
    side=np.median(data['R'][:min(20,len(t)),6]-data['R'][:min(20,len(t)),2],axis=0);side[1]=0;side/=np.linalg.norm(side)
    forward=np.cross([0.,1.,0.],side)
    return {'method':method,'window':window,'version':version,'title':title,'focus':focus,'description':description,
        'times':np.round(t,8).tolist(),'data':packed,'origin':origin.tolist(),'right':side.tolist(),'forward':forward.tolist(),
        'floor':float(data['R'][:,[5,9],1].min()),'metrics':stages,'source':str(path),'sha256':SHA(path),'frames':len(t),
        'rounding_max_error_m':max(float(np.max(abs(np.array(packed['raw'][k])-v))) for k,v in data.items())}


def main():
    cases=[]

    # Lead with the three discussed research issues, retain the rest of the fixed fit probes.
    order=sorted(SUMMARY['fits'],key=lambda r:0 if (r['method'],r['window_key'])==('motionlcm','031') else 1 if (r['method'],r['window_key'])==('ardy','051') else 2)
    for r in order:
        m,w=r['method'],r['window_key'];title=f'{m} · {w} · {Path(r["id"]).stem}'
        if w=='031':desc='躯干高误差窗口：观察原始姿态经过目标骨架转换后的倾斜变化。'
        elif m=='ardy' and w=='051':desc='ARDY 局部恶化：节点对应修正后，固定模板形状的偏差更加明显。'
        else:desc='本轮固定开发窗口；可对照躯干、四肢和足端位置。'
        cases.append(case(Path(r['output'])/'comparison.npz',m,w,'K3.3 转换；K3.1 拟合；耦合同步后评价端点不变',title,'torso',desc))
    # K3.2 arm candidate and K3.1 leg failure keep their own input/version lineage.
    cases.append(case(BASE/'evalkit_k32_continue_20261116/077/kimodo/comparison.npz','kimodo','077','K3.2 手臂连续性候选；旧转换','Kimodo · 077 · 擦拭手臂','right_arm','观察右臂时序连续性；此案例属于前轮 K3.2 候选。',key='F:k31'))
    cases.append(case(BASE/'evalkit_k31_full_20261116/074/kimodo/comparison.npz','kimodo','074','K3.1 原模型范围与锁定；旧转换','Kimodo · 074 · 跪坐左腿','left_leg','观察深屈膝与足部方向差异；展示原范围结果，扩范围实验单独报告。',key='F:k31'))
    cases=cases[:2]+cases[-2:]+cases[2:-2]
    payload={'cases':cases,'names':NAMES,'parents':PARENTS,'scope':'16 corrected-conversion fit windows + two explicitly versioned earlier cases; all development data',
     'display':'common 16 joint landmarks; no independent head/neck fit shown; raw world coordinates by default; normalized lengths come from original reference window'}
    template=(ROOT/'utils/motiogen/evalkit/visualization/motion_viewer.html').read_text()
    OUT.write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    manifest={'html':str(OUT),'sha256':SHA(OUT),'cases':[{k:v for k,v in c.items() if k not in ['data','times','origin','right','forward']} for c in cases],
     'preservation':'arrays read only; no fits, changes to frozen configs or holdout runs','frames_retained':sum(c['frames'] for c in cases)}
    (ROOT/'develop_logs/test_runs/20261117_k_animation/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(OUT,len(cases),OUT.stat().st_size)


if __name__ == "__main__":
    main()
