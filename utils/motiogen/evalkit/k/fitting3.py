"""Versioned fitting views, input-only projection and full-body support receipts."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
from .config import digest,implementation_key
from .motion import load
from .bones import project_motion,common,NAMES,PARENTS
from .geometry import unit


def write_coordinates(path,times,labels,coords):
    import opensim as osim
    table=osim.TimeSeriesTable();table.setColumnLabels(labels)
    for t,row in zip(times,coords):table.appendRow(float(t),osim.RowVector(row.tolist()))
    table.addTableMetaDataString('inDegrees','yes');osim.STOFileAdapter.write(table,str(path))


def positions(model_path,times,labels,coords):
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();result=[]
    for row in coords:
        biomech._set(model,state,labels,row)
        result.append([biomech._child_origin(model,state,j) for j,_,_ in biomech.MARKERS.values()])
    return np.asarray(result),[n for _,_,n in biomech.MARKERS.values()]


def interpolate(motion,query,names):
    points,pnames,_=motion.legacy();t=motion.times-motion.times[0]
    return np.stack([np.stack([np.interp(query,t,points[:,pnames.index(n),j]) for j in range(3)],1) for n in names],1)


def support(source,fitted,names):
    a,b=common(source,names),common(fitted,names);result={}
    regions={'pelvis':[2,6],'torso':[1,10,13],'left_arm':[11,12],'right_arm':[14,15],'left_leg':[3,4,5],'right_leg':[7,8,9]}
    for region,edges in regions.items():
        da=np.stack([unit(a[:,j]-a[:,PARENTS[j]]) for j in edges],1)
        db=np.stack([unit(b[:,j]-b[:,PARENTS[j]]) for j in edges],1)
        angles=np.degrees(np.arccos(np.clip((da*db).sum(-1),-1,1)))
        pos=np.linalg.norm(a[:,edges]-b[:,edges],axis=-1)
        observed=np.isfinite(angles).all(1)&np.isfinite(pos).all(1)
        result[region]={'observed':observed,'angle_valid':observed&(angles<=10).all(1),
                        'position_valid':observed&(pos<=.1).all(1),'angle_error_deg':angles,'position_error_m':pos}
    return result


def fit_configured(run,work,mass,floor,config):
    import biomech
    from .fitting import _fit_legacy
    work=Path(work);work.mkdir(parents=True,exist_ok=True);motion=load(run,config=config)
    settings=dict(config['fitting']);settings.update(settings.get('per_method',{}).get(motion.method,{}));view=motion;projection={'strategy':'none'}
    if settings.get('projection'):view,projection=project_motion(motion)
    key={'inputs':motion.provenance['inputs'],'config':config,'implementation':implementation_key(),
         'biomech':digest(biomech.__file__),'model':digest(biomech.MODEL),'mass':mass,'floor':floor}
    encoded=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    cache=work/'k3_receipt.json'
    if cache.exists() and json.loads(cache.read_text())['key']==encoded and (work/'kinematics.npz').exists():
        receipt=json.loads(cache.read_text())
        with np.load(work/'kinematics.npz') as z:
            times=z['times'];labels=list(z['labels']);coords=z['coordinates'];fitted=z['fitted'];names=list(z['names']);source=z['source'];projected=z['projected']
    else:
        base_work=Path(settings.get('base_cache',work/'base'))
        base=_fit_legacy(view,base_work,mass,floor)
        times,labels,coords=base['times'],base['labels'],base['coords'].copy()
        inside=(times>=0)&(times<=motion.times[-1]-motion.times[0]+1e-8)
        times,coords=times[inside],coords[inside]
        shutil.copy2(base_work/'model.osim',work/'model.osim')
        optimization={'strategy':'legacy'}
        if settings['strategy']=='direction_refinement' and motion.method in settings['methods']:
            from .refinement import refine
            coords,optimization=refine(work/'model.osim',view,times,labels,coords,settings)
        write_coordinates(work/'coordinates.mot',times,labels,coords)
        fitted,names=positions(work/'model.osim',times,labels,coords)
        source=interpolate(motion,times,names);projected=interpolate(view,times,names)
        receipt={'key':encoded,'evidence':key,'projection':projection,'optimization':optimization,'calibration':base['receipt']['calibration'],
                 'support_scope':'region_angle_valid is observed-input direction fidelity; region_registered_valid also requires upright registration; downstream arm/contact eligibility retains registration and position gates',
                 'fps':100.,'ik_marker_error':base['receipt']['ik_marker_error']}
        cache.write_text(json.dumps(receipt,indent=2)+'\n')
        np.savez_compressed(work/'kinematics.npz',times=times,labels=labels,coordinates=coords,names=names,fitted=fitted,source=source,projected=projected)
    quality=support(source,fitted,names)
    valid_cal=bool(receipt['calibration']['qualified']);angle_masks={k:v['angle_valid'] for k,v in quality.items()}
    spatial_masks={k:v['position_valid']&valid_cal for k,v in quality.items()}
    joint_error=np.linalg.norm(source-fitted,axis=-1).max(1)
    np.savez_compressed(work/'region_quality.npz',times=times,**{k+'_'+n:v for k,q in quality.items() for n,v in q.items()})
    return {'times':times,'labels':labels,'coords':coords,'receipt':receipt,'ik_max_error_m':joint_error,
            'calibration_valid':valid_cal,'position_fit_valid':joint_error<=.1,'unmeasurable':(joint_error>.1)|~np.full(len(times),valid_cal),
            'arm_valid':{s:angle_masks[p+'_arm']&spatial_masks[p+'_arm'] for s,p in [('l','left'),('r','right')]},
            'region_angle_valid':angle_masks,'region_registered_valid':{k:v&valid_cal for k,v in angle_masks.items()},'region_position_valid':spatial_masks}
