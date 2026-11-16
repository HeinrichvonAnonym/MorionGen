"""Original-reference scoring, fixed common masks, raw and fixed-length geometry."""
import numpy as np
from .bones import common,normalize,lengths,NAMES,PARENTS
from .geometry import unit


def stats(delta,angular=True):
    a=np.asarray(delta);finite=np.isfinite(a);v=a[finite]
    result={'valid':int(finite.sum()),'total':int(a.size),'coverage':float(finite.mean()) if a.size else 0.,
            'median_abs':float(np.median(abs(v))) if len(v) else None,'p95_abs':float(np.percentile(abs(v),95)) if len(v) else None,
            'signed_bias':float(v.mean()) if len(v) else None}
    if angular:result.update({f'within_{k}':float(np.mean(abs(v)<=k)) if len(v) else None for k in [5,10,20]})
    return result


def angles(p):
    x={n:p[:,i] for i,n in enumerate(NAMES)};v=np.broadcast_to([0.,1.,0.],x['pelvis_mid'].shape)
    angle=lambda a,b:np.degrees(np.arccos(np.clip((unit(a)*unit(b)).sum(-1),-1,1)))
    torso=x['shoulder_mid']-x['pelvis_mid'];out={'trunk_incline':angle(torso,v),
        'pelvis_line_incline':angle(x['right_hip']-x['left_hip'],v)}
    for s in ['left','right']:
        q=lambda n:x[s+'_'+n]
        out[s+'_elbow_flexion']=angle(q('elbow')-q('shoulder'),q('wrist')-q('elbow'))
        out[s+'_knee_flexion']=angle(q('knee')-q('hip'),q('ankle')-q('knee'))
        out[s+'_thigh_incline']=angle(q('hip')-q('knee'),v)
        out[s+'_upper_arm_elevation']=angle(q('elbow')-q('shoulder'),-v)
        out[s+'_hip_trunk_angle']=angle(q('hip')-q('knee'),torso)
        out[s+'_ankle_chain_angle']=angle(q('knee')-q('ankle'),q('foot')-q('ankle'))
        out[s+'_foot_incline']=angle(q('foot')-q('ankle'),v)
    return out


def region(name):
    if name.startswith('pelvis'):return 'pelvis'
    if name.startswith('trunk'):return 'torso'
    side='left' if name.startswith('left') else 'right'
    return side+('_arm' if 'elbow' in name or 'arm_' in name else '_leg')


def contact(points,times,floor=0.):
    result={}
    for side in ['left','right']:
        foot=points[:,NAMES.index(side+'_foot')];height=foot[:,1]-floor
        speed=np.linalg.norm(np.gradient(foot[:,[0,2]],times,axis=0),axis=-1)
        good=np.isfinite(height)&np.isfinite(speed);near=good&(height<=.03)
        result[side]={'valid_frames':int(good.sum()),'total_frames':len(times),'near_floor_share':float(np.mean(near)), 'below_floor_share':float(np.mean(good&(height<-.01))),
                      'speed_when_near_floor_mps':float(np.mean(speed[near])) if near.any() else None,
                      'height_p05_m':float(np.percentile(height[good],5)) if good.any() else None,
                      'scope':'ankle/toe joint landmark proxy on common input-derived plane; not force-validated contact'}
    return result


def compare(reference,degraded,projected,fits,times,supports,floor=0.,reference_lengths=None):
    """All arguments common-tree positions, all variants share physical times and masks."""
    raw={'R':reference,'D':degraded,**{f'P:{k}':v for k,v in projected.items()},**{f'F:{k}':v for k,v in fits.items()}}
    target=lengths(reference) if reference_lengths is None else np.asarray(reference_lengths,float)
    if target.shape!=(len(NAMES),):raise ValueError('Reference lengths must follow the common evaluation skeleton')
    norm={k:normalize(v,target) for k,v in raw.items()}
    result={};arrays={'times':times,'names':NAMES,'reference_lengths':target,**raw}
    for variant in fits:
        result[variant]={'raw':{},'normalized':{},'coverage':{},'contact':{k:contact(raw[k],times,floor) for k in ['R','D','P:'+variant,'F:'+variant]}}
    for space,points in [('raw',raw),('normalized',norm)]:
        values={k:angles(v) for k,v in points.items()}
        for variant in fits:
            stages={'degradation':('D','R'),'fit_deviation':('F:'+variant,'D'),'end_to_end':('F:'+variant,'R'),
                    'projection':('P:'+variant,'D'),'fit_after_projection':('F:'+variant,'P:'+variant)}
            summary={'angles':{},'positions':{},'directions':{}}
            for metric in values['R']:
                common_mask=np.logical_and.reduce([np.isfinite(v[metric]) for v in values.values()])
                summary['angles'][metric]={label:stats(np.where(common_mask,values[a][metric]-values[b][metric],np.nan)) for label,(a,b) in stages.items()}
                mask=common_mask&supports[variant].get(region(metric),np.zeros(len(times),bool))
                summary['angles'][metric]['supported_end_to_end']=stats(np.where(mask,values['F:'+variant][metric]-values['R'][metric],np.nan))
                arrays[variant+'_'+space+'_'+metric+'_e2e']=np.where(common_mask,values['F:'+variant][metric]-values['R'][metric],np.nan)
            for j,name in enumerate(NAMES):
                mask=np.logical_and.reduce([np.isfinite(p[:,j]).all(1) for p in points.values()])
                entry={}
                for label,(a,b) in stages.items():
                    diff=points[a][:,j]-points[b][:,j]
                    centred=diff-(points[a][:,0]-points[b][:,0])
                    entry[label]={'world':stats(np.where(mask,np.linalg.norm(diff,axis=-1),np.nan),False),
                                  'root_relative':stats(np.where(mask,np.linalg.norm(centred,axis=-1),np.nan),False),
                                  'vector_bias_m':np.nanmean(np.where(mask[:,None],diff,np.nan),axis=0).tolist() if mask.any() else None}
                summary['positions'][name]=entry
                if PARENTS[j]>=0:
                    vectors={k:unit(p[:,j]-p[:,PARENTS[j]]) for k,p in points.items()}
                    valid=np.logical_and.reduce([np.isfinite(v).all(1) for v in vectors.values()])
                    summary['directions'][name]={label:stats(np.where(valid,np.degrees(np.arccos(np.clip((vectors[a]*vectors[b]).sum(-1),-1,1))),np.nan)) for label,(a,b) in stages.items()}
            result[variant][space]=summary
    for variant in fits:
        result[variant]['coverage']={k:{'supported':int(v.sum()),'total':len(v),'share':float(v.mean())} for k,v in supports[variant].items()}
        result[variant]['length_component']={}
        # Exact vector difference in the same world frame, not subtraction of norms/P95.
        for label,a,b in [('degradation','D','R'),('fit_deviation','F:'+variant,'D'),('end_to_end','F:'+variant,'R')]:
            component=(raw[a]-raw[b])-(norm[a]-norm[b])
            result[variant]['length_component'][label]={n:stats(np.linalg.norm(component[:,j],axis=-1),False) for j,n in enumerate(NAMES)}
    return result,arrays
