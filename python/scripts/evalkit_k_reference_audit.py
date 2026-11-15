#!/usr/bin/env python3
"""Post-hoc R -> D -> F audit of saved development fits; never fits or tunes a model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
METRICS=['trunk_incline','left_elbow_flexion','right_elbow_flexion','left_knee_flexion','right_knee_flexion','left_thigh_incline','right_thigh_incline']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def summarize_signed(delta,total):
    delta=np.asarray(delta);good=np.isfinite(delta);d=delta[good]
    return {'valid_frames':int(good.sum()),'total_frames':int(total),'coverage':float(good.sum()/total) if total else 0.,
            'median_abs':float(np.median(abs(d))) if len(d) else None,'p95_abs':float(np.percentile(abs(d),95)) if len(d) else None,
            'signed_bias':float(d.mean()) if len(d) else None,
            'within_5':float(np.mean(abs(d)<=5)) if len(d) else None,
            'within_10':float(np.mean(abs(d)<=10)) if len(d) else None,
            'within_20':float(np.mean(abs(d)<=20)) if len(d) else None}


def decompose(reference,degraded,fitted,valid):
    """Same frames/definitions for all three terms; quantiles are not additive."""
    common=np.asarray(valid,bool)&np.isfinite(reference)&np.isfinite(degraded)&np.isfinite(fitted)
    d=np.where(common,degraded-reference,np.nan)
    k=np.where(common,fitted-degraded,np.nan)
    e=np.where(common,fitted-reference,np.nan)
    return d,k,e,common


def resample_points(times,points,query):
    # Only interpolate inside the recorded clock; NaN bracketing stays invalid.
    out=np.stack([np.interp(query,times,p) for p in points.reshape(len(times),-1).T],1).reshape(len(query),*points.shape[1:])
    out[(query<times[0]-1e-8)|(query>times[-1]+1e-8)]=np.nan
    return out


def fit_positions(work):
    import opensim as osim
    import biomech
    times,labels,coords=biomech.read_storage(work/'coordinates.mot')
    model=osim.Model(str(work/'model.osim'));state=model.initSystem();positions=[]
    for row in coords:
        biomech._set(model,state,labels,row)
        positions.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
    return times,np.asarray(positions)


def one(args):
    cohort,rows,data,out=args
    import opensim as osim
    osim.Logger.setLevelString('error')
    import biomech
    from motion_retarget.bvh import parse_bvh
    from motiogen.evalkit.k.reference import archive
    from motiogen.evalkit.k.export import ALIASES
    from motiogen.evalkit.k import load,Motion
    from motiogen.evalkit.k.body import evaluate_body
    names=[v[2] for v in biomech.MARKERS.values()]
    base=rows[0];src=base['source_bvh'];root=Path(read(ROOT/'config/gold_k_20261115/evidence.json')['local_roots']['collaborate'])
    zipfile=archive(str(root/src['archive']));info=zipfile.getinfo(src['member'])
    assert info.CRC==int(src['crc32'],16) and info.file_size==src['bytes']
    b=parse_bvh(zipfile.read(src['member']));positions,_=b.forward_kinematics(True)
    sn={ALIASES.get(j.name,j.name):i for i,j in enumerate(b.joints)}
    start,end=[int(round(t*b.fps)) for t in base['source_window_s']]
    raw=positions[start:end+1]*.01
    rp=np.stack([raw[:,sn[n]] if n in sn else np.full((len(raw),3),np.nan) for n in names],1)
    rt=np.arange(len(rp))/b.fps
    outputs=[]
    for row in rows:
        assert row['partition']=='development' and row['source_window_s']==base['source_window_s']
        source=Path(row['source']);input_file=source/'motion.npz' if source.is_dir() else source;motion=load(source,row['method']);dp,dn,_=motion.legacy();dt=motion.times-motion.times[0]
        provenance_file=input_file.parent/'k_provenance.json'
        prior={ALIASES.get(n,n) for n in read(provenance_file)['prior_joints']}
        scored_names={s+'_'+j for s in ['left','right'] for j in ['shoulder','elbow','wrist','hip','knee','ankle']}
        assert not prior.intersection(scored_names), 'Scored body chain contains template priors'
        dp=np.stack([dp[:,dn.index(n)] if n in dn else np.full((len(dp),3),np.nan) for n in names],1)
        works={v:data/cohort/folder/row['key']/'work' for v,folder in [('k1','k1'),('k2','k2_adaptive')]}
        evidence={'input_sha256':digest(input_file),'provenance_sha256':digest(provenance_file),
                  'manifest_sha256':digest(data/cohort/'manifest.json'), 'scored_chains_have_no_template_priors':True,
                  'fits':{v:{f:digest(w/f) for f in ['model.osim','coordinates.mot','receipt.json']} for v,w in works.items()}}
        fits={v:fit_positions(w) for v,w in works.items()}
        query=fits['k2'][0];in_clock=(query>=max(rt[0],dt[0],fits['k1'][0][0])-1e-8)&(query<=min(rt[-1],dt[-1],fits['k1'][0][-1])+1e-8)
        original_count=len(query);query=query[in_clock]
        points={'reference':resample_points(rt,rp,query),'degraded':resample_points(dt,dp,query),
                **{v:resample_points(t,p,query) for v,(t,p) in fits.items()}}
        angles={k:evaluate_body(Motion(p,names,query,k)) for k,p in points.items()}
        rec={k:row[k] for k in ['id','group_id','method','key','source_window_s','source_bvh']}
        rec.update(cohort=cohort,frames=len(query),excluded_clock_frames=original_count-len(query),evidence=evidence,metrics={},positions={})
        arrays={'times':query,'joint_names':names,**{k+'_points':p for k,p in points.items()}}
        for metric in METRICS:
            reference=angles['reference'][metric];degraded=angles['degraded'][metric]
            valid=reference['valid']&degraded['valid']&angles['k1'][metric]['valid']&angles['k2'][metric]['valid']
            rec['metrics'][metric]={}
            for version in ['k1','k2']:
                d,k,e,common=decompose(reference['values_deg'],degraded['values_deg'],angles[version][metric]['values_deg'],valid)
                assert np.nanmax(abs(e-(d+k)))<1e-9 if common.any() else True
                receipt=read(works[version]/'receipt.json')
                # Reproduce each version's existing support declaration, distinct from reference accuracy.
                if version=='k2' and 'elbow' in metric:
                    side='l' if metric.startswith('left') else 'r'
                    with np.load(works[version]/'arm_quality.npz') as z:
                        support=np.interp(query,z['times'],z[side+'_valid'].astype(float))>=1.-1e-9
                    support &= receipt['calibration']['qualified']
                else:
                    _,el,err=biomech.read_storage(next((works[version]/'ik').glob('*_ik_marker_errors.sto')))
                    indices=np.clip(np.rint(query*receipt['fps']).astype(int),0,len(err)-1)
                    support=(err[indices,el.index('marker_error_max')]<=.1)&receipt['calibration']['qualified']
                support &= common
                rec['metrics'][metric][version]={'degradation':summarize_signed(d,len(query)),
                    'evalkit_added':summarize_signed(k,len(query)),'end_to_end':summarize_signed(e,len(query)),
                    'declared_support_frames':int(support.sum()),'end_to_end_on_declared_support':summarize_signed(e[support],len(query)),
                    'support_scope':'K1 global marker/calibration gate; K2 elbow source-fidelity/calibration gate; neither gate verifies original-reference accuracy',
                    'cancellation_share':float(np.mean(d[common]*k[common]<0)) if common.any() else None}
                arrays[metric+'_'+version+'_degradation']=d;arrays[metric+'_'+version+'_added']=k;arrays[metric+'_'+version+'_end_to_end']=e
                arrays[metric+'_'+version+'_support']=support
        for side in ['left','right']:
            index=names.index(side+'_wrist');rec['positions'][side]={}
            for version in ['k1','k2']:
                r,d,f=(points[n][:,index] for n in ['reference','degraded',version])
                valid=np.isfinite(r).all(1)&np.isfinite(d).all(1)&np.isfinite(f).all(1)
                rec['positions'][side][version]={}
                for label,diff in [('degradation',d-r),('evalkit_added',f-d),('end_to_end',f-r)]:
                    norms=np.linalg.norm(diff[valid],axis=-1)
                    rec['positions'][side][version][label]={'valid_frames':int(valid.sum()),'total_frames':len(query),
                        'median_m':float(np.median(norms)) if len(norms) else None,'p95_m':float(np.percentile(norms,95)) if len(norms) else None,
                        'vector_bias_m':diff[valid].mean(0).tolist() if valid.any() else None}
        file=out/(hashlib.sha256((cohort+'|'+row['id']+'|'+row['method']).encode()).hexdigest()[:20]+'.npz')
        np.savez_compressed(file,**arrays);rec['arrays']=str(file);outputs.append(rec)
        assert evidence['input_sha256']==digest(input_file)
        assert evidence['provenance_sha256']==digest(provenance_file)
        for v,w in works.items():
            assert all(digest(w/f)==sha for f,sha in evidence['fits'][v].items()),'Saved fit changed during read-only audit'
    return outputs


def aggregate(rows):
    out={}
    for method in sorted(set(r['method'] for r in rows)):
        selected=[r for r in rows if r['method']==method];out[method]={}
        for version in ['k1','k2']:
            result={}
            for stage in ['degradation','evalkit_added','end_to_end']:
                worst=[max(r['metrics'][s+'_elbow_flexion'][version][stage]['p95_abs'] for s in ['left','right']) for r in selected]
                result[stage]={'record_macro_worse_side_p95_deg':float(np.mean(worst)),
                               'recordings':len(selected),'groups':len(set(r['group_id'] for r in selected))}
            out[method][version]=result
        regressions=[];reversals=[]
        for r in selected:
            error=lambda v,s:max(r['metrics'][side+'_elbow_flexion'][v][s]['p95_abs'] for side in ['left','right'])
            d=error('k2','end_to_end')-error('k1','end_to_end')
            if d>1:regressions.append({'id':r['id'],'delta_p95_deg':d})
            if error('k2','evalkit_added')<error('k1','evalkit_added') and d>0:
                reversals.append({'id':r['id'],'end_to_end_delta_p95_deg':d})
        out[method]['end_to_end_regressions_over_1_deg']=regressions
        out[method]['fidelity_improved_but_end_to_end_worsened']=reversals
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);tasks=[]
    for cohort in ['development','development_first']:
        rows=read(a.data/cohort/'manifest.json')['records']
        for id in sorted(set(r['id'] for r in rows)):tasks.append((cohort,[r for r in rows if r['id']==id],a.data,a.out))
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        groups=list(pool.map(one,tasks))
    rows=[r for group in groups for r in group]
    result={'scope':'Read-only post-hoc audit of 22 original development BVH windows, 66 controlled exports and existing K1/K2 fits; no IK rerun, tuning or final holdout reuse',
            'definitions':{'R':'Original BVH positions in metres, Y-up, before filtering/retargeting; original window reconstructed from manifest',
                           'D':'Saved serialized degraded input decoded to common joint positions; includes temporal, rig and codec losses',
                           'F':'Saved model plus filtered coordinates, evaluated by FK at final 100 Hz clock',
                           'degradation':'metric(D)-metric(R)','evalkit_added':'metric(F)-metric(D)','end_to_end':'metric(F)-metric(R)',
                           'timing':'Same physical times and common finite mask for R/D/K1/K2; interpolate positions, then derive angles; no new smoothing or alignment',
                           'position':'Absolute world-space wrist positions; target-body shape changes included; no scale normalization',
                           'reference_quality':'Published BVH geometric reference; external anatomical accuracy not established',
                           'summary':'Within each record use worse-side P95; average records equally. Signed scalar errors add on identical frames; norms and percentiles do not.'},
            'script_sha256':digest(__file__),'aggregate':aggregate(rows),'records':rows}
    a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['aggregate'],indent=2))


if __name__=='__main__':main()
