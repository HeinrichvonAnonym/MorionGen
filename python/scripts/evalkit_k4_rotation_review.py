"""Independent original-R acceptance and native-vs-interpolation attribution."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from scipy.spatial.transform import Rotation
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.scoring import summary,unit,DIRECTIONS
from motiogen.evalkit.k.strict_acceptance import assess
from motiogen.degradation.metrics import points,normalize,angle
from evalkit_k4_ablation_score import at
from evalkit_k4_score import serializable


def read(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def local_path(native,output,audit):
    """Sampled path excess is geometric diagnostic, not reference accuracy."""
    result={};t=native['times'];new=output['times'];labels=list(native['labels'])
    for name,row in audit['joints'].items():
        if row['classification']!='three_axis_so3':continue
        cols=[labels.index(n) for n in row['rotation_coordinates']];signs=np.array(row['signs'])
        nr=Rotation.from_euler(row['order'],native['coords'][:,cols]*np.pi/180*signs)
        pr=Rotation.from_euler(row['order'],output['coords'][:,cols]*np.pi/180*signs)
        direct=np.rad2deg((nr[:-1].inv()*nr[1:]).magnitude());excess=[]
        for i in range(len(t)-1):
            inside=(new>t[i]+1e-10)&(new<t[i+1]-1e-10)
            rr=Rotation.from_quat(np.concatenate([nr[i:i+1].as_quat(),pr[inside].as_quat(),nr[i+1:i+2].as_quat()]))
            travel=np.rad2deg((rr[:-1].inv()*rr[1:]).magnitude()).sum()
            excess.append(max(0,float(travel-direct[i])))
        result[name]={'native_endpoint_step_deg':summary(direct),'sampled_path_excess_deg':summary(excess),
                      'intervals_excess_gt_1deg':int(np.sum(np.array(excess)>1))}
    return result


def attribution(raw_reference,native,output):
    clock=native.times;r=at(raw_reference,clock);p=at(output,clock)
    rr=points(r,True);nn=points(native);pp=points(p)
    lengths=points(raw_reference,True);norms=[normalize(lengths,v) for v in (rr,nn,pp)]
    result={}
    for geometry,(a,b,c) in [('raw',(rr,nn,pp)),('common_length',norms)]:
        block={};closure=[]
        for name in a:
            if name not in b or name not in c:continue
            fit=b[name]-a[name];extra=c[name]-b[name];total=c[name]-a[name]
            finite=np.isfinite(fit+extra+total).all(-1)
            if finite.any():closure.append(float(abs(total-fit-extra)[finite].max()))
            block[name]={'native_minus_R_cm':summary(np.linalg.norm(fit,axis=-1)*100),
                         'export_minus_native_cm':summary(np.linalg.norm(extra,axis=-1)*100),
                         'export_minus_R_cm':summary(np.linalg.norm(total,axis=-1)*100)}
        result[geometry]={'positions':block,'vector_identity_max_abs_m':max(closure,default=None)}
    directions={}
    for name,(start,end) in {**DIRECTIONS,'pelvis_line':('left_hip','right_hip')}.items():
        a=nn[end]-nn[start];b=pp[end]-pp[start]
        rate=np.diff(unit(b)-unit(a),axis=0)/np.diff(clock)[:,None]
        directions[name]={'export_minus_native_deg':summary(angle(a,b)),
                          'export_minus_native_direction_rate_deg_s':summary(np.linalg.norm(rate,axis=-1)*180/np.pi)}
    result['directions']=directions
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Immutable review exists')
    protocol=json.loads((a.run/'protocol.json').read_text());policy=json.loads((ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_text());rows=[]
    for item in protocol['records']:
        w,m=item['window'],item['method'];row={'window':w,'method':m,'source_fit':item['fit'],'conditions':{}};rows.append(row)
        replay=a.run/'replays'/w/m;receipt_path=replay/'receipt.json'
        if not receipt_path.exists():row['status']='replay_failed';continue
        receipt=json.loads(receipt_path.read_text());row['resampling']=receipt['whole_body']
        ndata=read(replay/'views/native'/w/m/'fk.npz');native=Motion(ndata['positions'],list(ndata['names']),ndata['times'],'native_fk')
        nq=read(replay/'views/native'/w/m/'coordinates.npz')
        native_score=json.loads((a.run/'scores/native'/f'{w}_{m}.json').read_text())
        baseline_file=Path(protocol['baseline'])/'scores'/f'{w}_{m}.json'
        frozen_baseline=json.loads(baseline_file.read_text())
        with np.load(native_score['original_reference_file']) as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
        for condition in protocol['conditions']:
            file=a.run/'scores'/condition/f'{w}_{m}.json'
            if not file.exists():
                row['conditions'][condition]={'status':'score_failed','reason':receipt.get('legacy_control_failures',{}).get(condition,'missing score')}
                continue
            score=json.loads(file.read_text());data=read(replay/'views'/condition/w/m/'fk.npz');motion=Motion(data['positions'],list(data['names']),data['times'],condition)
            values={'acceptance_frozen_policy':assess(score,frozen_baseline,policy),'temporal_baseline_file':str(baseline_file),'acceptance_relative_native':assess(score,native_score,policy),'acceptance_relative_D':assess(score,None,policy),
                    'score_file':str(file),'full_comparison_summaries':{},'continuous_comparison_summaries':{},'original_clock_comparison_summaries':{}}
            for stratum,field in [('full','full_comparison_summaries'),('continuous','continuous_comparison_summaries')]:
                for pair in ('D-R','F-D','F-R'):
                    block=score['comparisons'][pair][stratum]
                    values[field][pair]={'directions_deg':{k:v['p95'] for k,v in block['directions_deg'].items()},
                        'joint_angles_deg':{k:v['p95'] for k,v in block['joint_angles_deg'].items()},
                        'positions_cm':{n:v['world']['p95'] for region in block['positions_cm'].values() for n,v in region.items()},
                        'common_length_positions_cm':{n:v['common_length_world']['p95'] for region in block['positions_cm'].values() for n,v in region.items()},
                        'head_relative_thorax_deg':score['head_neck_rotation_comparisons'][pair][stratum]['p95']}
            values['original_clock_comparison_summaries']=score['original_clock']['comparisons']
            values['attribution']=attribution(r,native,motion)
            q=read(replay/'views'/condition/w/m/'coordinates.npz');values['joint_path']=local_path(nq,q,receipt['whole_body']['audit'])
            row['conditions'][condition]=values
        row['status']='completed' if all('acceptance_relative_native' in row['conditions'].get(c,{}) for c in protocol['conditions'] if c not in protocol.get('optional_controls',[])) else 'incomplete'
        row['control_failures']=receipt.get('legacy_control_failures',{})
    result={'expected':len(protocol['records']),'completed':sum(r['status']=='completed' for r in rows),'rows':rows,
        'attribution_definition':'N=native model FK at D timestamps under shared root/head definition. E=exported FK, then position readback to D timestamps using unchanged scorer. E-R=(N-R)+(E-N) in vectors only; no subtraction of norms/P95/rotation distances. Full score files include original R clock and frozen jump strata.',
        'path_definition':'Sampled local joint SO(3) arc length minus native endpoint geodesic; path difference is not a standalone physical accuracy measure.',
        'scope':'Paired interpolation experiment, no refitting or geometric changes. Six v6 fits and fourteen older common fits are separate source strata. Original-R targets remain independent.',
        'promotion':'none; explicit candidate only','new_holdout_uses':0,
        'review_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest(),
        'policy_sha256':hashlib.sha256((ROOT/'config/evalkit_k4_20261117/strict_v1/acceptance.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
    print('completed',result['completed'],'/',result['expected'])


if __name__=='__main__':main()
