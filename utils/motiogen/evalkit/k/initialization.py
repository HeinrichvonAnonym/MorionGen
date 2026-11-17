"""Input-only, clip-level IK multistart with fixed model geometry and joint domain."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def flexion(hip, knee, ankle):
    a,b=knee-hip,ankle-knee
    value=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))
    valid=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)>1e-12
    return np.where(valid,value,np.nan)


def structure(path):
    tree=ET.parse(path)
    for coordinate in tree.findall('.//Coordinate'):
        value=coordinate.find('default_value')
        if value is not None:value.text='INITIALIZATION_ONLY'
    return ET.tostring(tree.getroot())


def seed_definitions(model_path,points,names):
    """First-frame observed hip line and knee geometry; missing seeds stay unavailable."""
    import opensim as osim
    import biomech
    from .k4_fitting import model_fk
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    defaults=np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    idx={n:i for i,n in enumerate(names)};seeds={'nominal':{}};scans={};knees={};unavailable={}
    for side,s in [('left','l'),('right','r')]:
        target=flexion(*[points[0,idx[side+'_'+j]] for j in ('hip','knee','ankle')])
        c=cs.get('knee_angle_'+s)
        if not np.isfinite(target) or c.getDefaultLocked() or c.isDependent(state):
            unavailable[side+'_knee']='missing/degenerate first-frame input or fixed model coordinate';continue
        grid=np.linspace(c.getRangeMin(),c.getRangeMax(),281)*180/np.pi
        q=np.tile(defaults,(len(grid),1));q[:,labels.index(c.getName())]=grid
        fk,fn=model_fk(model_path,labels,q,direct_couplers=True)
        angles=flexion(*[fk[:,fn.index(side+'_'+j)] for j in ('hip','knee','ankle')])
        chosen=int(np.nanargmin(abs(angles-target)));knees[c.getName()]=float(grid[chosen])
        scans[side]={'input_initial_flexion_deg':float(target),'coordinate_deg':grid.tolist(),
            'geometric_flexion_deg':angles.tolist(),'seed_deg':float(grid[chosen]),
            'seed_mapping_error_deg':float(abs(angles[chosen]-target))}
    if len(knees)==2:seeds['knee']=knees
    h=points[0,idx['right_hip']]-points[0,idx['left_hip']]
    neutral=biomech._child_origin(model,state,'hip_r')-biomech._child_origin(model,state,'hip_l')
    if np.isfinite(h).all() and np.linalg.norm(h[[0,2]])>=1e-6:
        yaw=np.arctan2(h[0],h[2])-np.arctan2(neutral[0],neutral[2])
        yaw=float(np.rad2deg(np.arctan2(np.sin(yaw),np.cos(yaw))))
        seeds['heading']={'pelvis_rotation':yaw}
        if 'knee' in seeds:seeds['heading_knee']={**knees,'pelvis_rotation':yaw}
    else:unavailable['heading']='horizontal first-frame hip line missing or degenerate'
    return seeds,{'knee_mapping':scans,'unavailable':unavailable,
        'source':'D first-frame joint centres and fixed model coordinate-to-geometry mapping',
        'grid':'281 samples within declared current-model knee coordinate range; seed only, final IK is continuous'}


def weighted_marker_loss(model_path,labels,coordinates,points,names,weights):
    """Exact declared marker positions, input-only common full-window mean square."""
    import opensim as osim
    import biomech
    from .constraints import couplers,synchronize
    model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    selected=[name for name,w in weights.items() if w>0]
    sources={n:v[2] for n,v in biomech.MARKERS.items()}
    sources.update({n:v[1] for n,v in biomech.AXIAL.items()})
    target=points[:,[names.index(sources[n]) for n in selected]]
    if len(coordinates)!=len(target) or not np.isfinite(target).all():
        raise ValueError('Multistart requires the same complete observed marker window for every candidate')
    markers=[model.getMarkerSet().get(n) for n in selected];ws=np.array([weights[n] for n in selected])
    errors=[]
    for q,goal in zip(coordinates,target):
        biomech._set(model,state,labels,q);synchronize(state,bindings);model.realizePosition(state)
        p=np.array([[m.getLocationInGround(state).get(j) for j in range(3)] for m in markers])
        errors.append(np.sum(ws*np.sum((p-goal)**2,axis=-1))/ws.sum())
    errors=np.asarray(errors)
    if not np.isfinite(errors).all():raise ValueError('Nonfinite candidate residual')
    return {'mean_weighted_squared_marker_error_m2':float(errors.mean()),
            'weighted_marker_rms_p95_cm':float(np.percentile(np.sqrt(errors)*100,95)),
            'frames':len(errors),'markers':selected,'weights':weights}


def choose(candidates):
    valid=[(name,row) for name,row in candidates.items() if row.get('status')=='completed']
    if not valid:raise ValueError('All input-only IK initializations failed')
    return min(valid,key=lambda item:item[1]['input_loss']['mean_weighted_squared_marker_error_m2'])[0]


def run(model_path,trc,points,names,times,weights,work,settings):
    import opensim as osim
    import biomech
    from .timebase import native_clock
    if settings.get('strategy')!='input_multistart':raise ValueError('Unsupported initialization strategy')
    if settings.get('selection')!='full_window_weighted_markers':raise ValueError('Unsupported initialization selection')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    model_path=Path(model_path);original_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    seeds,definition=seed_definitions(model_path,points,names)
    order=settings.get('seeds',['nominal','knee','heading','heading_knee'])
    if len(set(order))!=len(order) or 'nominal' not in order or set(order)-{'nominal','knee','heading','heading_knee'}:
        raise ValueError('Initialization requires unique declared seeds including nominal')
    candidates={};duration=float(times[-1]-times[0])
    for name in order:
        if name not in seeds:
            candidates[name]={'status':'unavailable','reason':definition['unavailable']};continue
        folder=work/name;folder.mkdir();(folder/'ik').mkdir()
        probe=osim.Model(str(model_path))
        for coordinate,value in seeds[name].items():probe.updCoordinateSet().get(coordinate).setDefaultValue(np.deg2rad(value))
        path=folder/'model.osim';probe.printToXML(str(path))
        if structure(path)!=structure(model_path):raise ValueError('Model changed beyond initial coordinate defaults')
        row={'defaults_deg':seeds[name],'status':'failed'}
        try:
            biomech.inverse_kinematics(path,trc,folder/'ik/ik.mot',0.,duration,weights)
            clock,labels,q=biomech.read_storage(folder/'ik/ik.mot');_,clock_info=native_clock(clock,times)
            loss=weighted_marker_loss(path,labels,q,points,names,weights)
            row.update(status='completed',input_loss=loss,clock=clock_info)
        except Exception as error:row['error']=str(error)
        candidates[name]=row
    selected=choose(candidates)
    receipt={'definition':definition,'candidates':candidates,'selected':selected,'settings':settings,
        'source_model_sha256':original_hash,'selection':'single full-clip candidate by input-only weighted marker loss; no frame switching',
        'scope':'initial state only; model geometry, domains, weights and observations fixed; no R access'}
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=original_hash:raise ValueError('Source model changed')
    (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return work/selected/'ik',receipt
