"""Explicit K4 fitting entry. Only D and declared model/configuration are read."""
import json
from pathlib import Path
import numpy as np
from .config import ROOT,digest
from .motion import load
from .fixed_geometry import build


def model_fk(model_path,labels,coordinates,assemble=True,return_coordinates=False):
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();points=[];consistent=[]
    for row in coordinates:
        biomech._set(model,state,labels,row)
        if assemble:model.assemble(state)
        model.realizePosition(state)
        points.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
        consistent.append([model.getCoordinateSet().get(name).getValue(state)*(180/np.pi if model.getCoordinateSet().get(name).getMotionType()==1 else 1) for name in labels])
    result=(np.asarray(points),[source for _,_,source in biomech.MARKERS.values()])
    return (*result,np.asarray(consistent)) if return_coordinates else result


def write_coordinates(path,times,labels,coordinates):
    import opensim as osim
    table=osim.TimeSeriesTable();table.setColumnLabels(list(labels))
    for t,row in zip(times,coordinates):table.appendRow(float(t),osim.RowVector(row.tolist()))
    table.addTableMetaDataString('inDegrees','yes');osim.STOFileAdapter.write(table,str(path))


def fit(run,work,mass,floor,config):
    import biomech
    settings=json.loads(Path(config).read_text()) if not isinstance(config,dict) else dict(config)
    if settings.get('version')!='k4-fit-1':raise ValueError('Unsupported K4 fit configuration')
    motion=load(run,settings.get('method'),config=ROOT/'config/evalkit_k4_20261117/input.json')
    out=Path(work);out.mkdir(parents=True,exist_ok=True)
    key={'inputs':motion.provenance['inputs'],'config':settings,'mass_kg':mass,'floor_m':floor,
         'model_sha256':digest(biomech.MODEL),'implementation':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
         'input_config_sha256':digest(ROOT/'config/evalkit_k4_20261117/input.json'),
         'biomech_sha256':digest(biomech.__file__)}
    receipt_path=out/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['key']!=key:raise ValueError('K4 cache identity changed; use a new output directory')
        with np.load(out/'fit.npz') as z:return {**{k:z[k] for k in z.files},'receipt':receipt}
    geometry=build(motion,out/'model.osim',mass,settings['independent_geometry'])
    points,names,fps=motion.legacy()
    biomech.write_trc(points,names,fps,out/'markers.trc')
    weights={n:w for n,w in biomech.WEIGHTS.items() if n in biomech.MARKERS or n in geometry['registered_axial_markers']}
    (out/'ik').mkdir(exist_ok=True)
    biomech.inverse_kinematics(out/'model.osim',out/'markers.trc',out/'ik/ik.mot',0.,(len(points)-1)/fps,weights)
    times,labels,coordinates=biomech.write_coordinates(out/'ik/ik.mot',out/'coordinates.mot')
    fitted,fitted_names,consistent=model_fk(out/'model.osim',labels,coordinates,return_coordinates=True)
    write_coordinates(out/'coordinates_filtered_unassembled.mot',times,labels,coordinates)
    write_coordinates(out/'coordinates.mot',times,labels,consistent)
    values={'times':times+motion.times[0],'labels':np.asarray(labels),'coords':consistent,
            'coordinates_filtered_unassembled':coordinates,
            'positions':fitted,'names':np.asarray(fitted_names),'position_finite':np.isfinite(fitted).all(-1)}
    np.savez_compressed(out/'fit.npz',**values)
    receipt={'key':key,'version':settings['version'],'geometry':geometry,
             'scope':'constraint-assembled model FK; no independent head joint; native rotations reported separately',
             'kinematic_only':True,'force_eligible':False,'clock_origin_s':float(motion.times[0]),
             'input_rig_status':motion.provenance['input_geometry'],
             'coordinate_contract':'coordinates.mot contains constraint-assembled coordinates; original filtered coordinates preserved separately'}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return {**values,'receipt':receipt}
