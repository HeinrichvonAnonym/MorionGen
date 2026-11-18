"""Build an offline viewer from immutable same-native rotation replays and scores."""
import argparse,base64,gzip,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.metrics import points,angle
from evalkit_k4_ablation_score import at,jump_mask


def finite(x):return np.where(np.isfinite(x),np.round(x,6),None).tolist()

def read(path):
    with np.load(path) as z:return Motion(z['positions'],list(z['names']),z['times'],'fk')

def compact(block):
    if isinstance(block,dict):
        if 'p95' in block and 'n' in block:return {k:block[k] for k in ('p95','n','denominator') if k in block}
        return {k:compact(v) for k,v in block.items() if k not in ('temporal','world_bias_xyz_cm','pelvis_relative','common_length_world','common_length_pelvis_relative')}
    return block


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--coordinated',action='store_true',help='Show same-native SO3 versus coordinated FK on original R clock');a=p.parse_args()
    if a.output.exists():raise ValueError('Keep historical visualization snapshots immutable')
    frozen=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    artifacts={(r['window'],r['method']):r for r in json.loads((frozen/'artifact_verification.json').read_text())['records']};cases=[]
    for run in a.runs:
        protocol=json.loads((run/'protocol.json').read_text());reviewpath=run/('review.json' if a.coordinated else 'review_final.json');review=json.loads(reviewpath.read_text());lookup={(r['window'],r['method']):r for r in review['rows']}
        for row in protocol['records']:
            w,m=row['window'],row['method'];source=frozen/'inputs'/w/m/'candidate';d=load(source/'factorial/reference.npz' if m=='primal' else source,m,config=ROOT/'config/evalkit_k4_20261117/input.json')
            refpath=frozen/'references'/f'{w}.npz';r=read(refpath);rv=lookup[w,m];replay=run/'replays'/w/m
            motions={'R':r,'D':d};files={'R':str(refpath),'D':str(source)};scores={};acceptance={}
            roles=[('N','native'),('L','whole_native_grid'),('S','coordinated')] if a.coordinated else [('N','native'),('L','linear'),('H','hip_slerp'),('S','whole_so3')]
            for key,condition in roles:
                file=replay/'views'/condition/w/m/'fk.npz'
                if not file.exists():continue
                motions[key]=read(file);files[key]=str(file);scorefile=run/'scores'/condition/f'{w}_{m}.json';s=json.loads(scorefile.read_text())
                if a.coordinated:s=s['original_clock']
                scores[key]={stratum:{pair:compact(s['comparisons'][pair][stratum]) for pair in ['D-R','F-D','F-R']} for stratum in ['continuous','full','source_jump_tolerance']}
                for stratum in scores[key]:
                    for pair in scores[key][stratum]:scores[key][stratum][pair]['head_relative_thorax_deg']=compact(s['head_neck_rotation_comparisons'][pair][stratum])
                acceptance[key]=rv['conditions'][condition]['acceptance_original_clock_relative_D' if a.coordinated else 'acceptance_frozen_policy']
            clock=r.times if a.coordinated else d.times
            synced={k:points(at(v,clock),k=='R') for k,v in motions.items()};names=list(synced['D']);series={k:{'times':clock.tolist(),'positions':finite(np.stack([v[n] for n in names],1))} for k,v in synced.items()};traces={}
            for name,(start,end) in {'left_foot_axis':('left_ankle','left_foot'),'pelvis_line':('left_hip','right_hip'),'right_upper_arm':('right_shoulder','right_elbow'),'trunk_axis':('pelvis_mid','shoulder_mid')}.items():
                traces[name]={k:finite(angle(synced['R'][end]-synced['R'][start],v[end]-v[start])) for k,v in synced.items() if k!='R'}
            traces['left_foot_increment']={k:finite(np.linalg.norm(v['left_foot']-synced['N']['left_foot'],axis=1)*100) for k,v in synced.items() if k not in ['R','D']}
            peak=int(np.nanargmax(np.linalg.norm(synced['L']['left_foot']-synced['N']['left_foot'],axis=1)))
            cases.append(dict(window=w,method=m,source='固定原生拟合／协调路径' if a.coordinated else ('v6 rotation_only' if 'sentinels_v6' in row['fit'] else '旧 common_v1'),panel='补充诊断' if 'stress' in str(run) else '固定面板',series=series,names=names,score_times=clock.tolist(),tolerance=jump_mask(clock,d,artifacts[w,m]['source_jump_events']).tolist(),traces=traces,scores=scores,acceptance=acceptance,peak=peak,fallback=rv['resampling']['fallback_intervals'],control_failures=rv.get('control_failures',{}),evidence={'files':files,'source_fit':row['fit'],'review':str(reviewpath),'protocol_sha256':hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest(),'model_sha256':row['model_sha256']}))
    priority=[('074','kimodo'),('070','chois'),('015','kimodo'),('017','chois'),('000','kimodo'),('022','kimodo'),('069','kimodo')]
    cases.sort(key=lambda c:priority.index((c['window'],c['method'])) if (c['window'],c['method']) in priority else 100+int(c['window']))
    pairs=[('pelvis_mid','shoulder_mid'),('left_hip','right_hip'),('left_shoulder','right_shoulder'),('shoulder_mid','neck'),('neck','head')]
    for side in ['left','right']:
        for chain in [('hip','knee','ankle','foot'),('shoulder','elbow','wrist')]:pairs.extend((side+'_'+x,side+'_'+y) for x,y in zip(chain[:-1],chain[1:]))
    data=json.dumps(dict(cases=cases,edges=pairs,version='20261118 / 固定原生协调路径' if a.coordinated else '20261118 / SO(3) 全身重放',scope='局部路径候选；整体 K 尚未达标' if a.coordinated else '24 组主要对照；旧髋对照 1 组失败保留；整体 K 尚未达标',coordinated=a.coordinated),ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
    payload=base64.b64encode(gzip.compress(data.encode(),mtime=0)).decode()
    template=(ROOT/'utils/motiogen/evalkit/visualization/k4_rotation_viewer.html').read_text();a.output.write_text(template.replace('__DATA__',payload));print(a.output,'uncompressed',len(data),'embedded',len(payload),'cases',len(cases))


if __name__=='__main__':main()
