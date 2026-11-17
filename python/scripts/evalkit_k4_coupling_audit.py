"""Numerical joint-function and FK audit of an explicitly configured model copy."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.diagnostic_domain import apply
from motiogen.evalkit.k.constraints import couplers,synchronize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','config','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import shutil
    modelpath=a.output/'model.osim';shutil.copyfile(a.model,modelpath)
    config=json.loads(a.config.read_text());domain=apply(modelpath,config['kinematic_domain'])
    extension=None
    if config.get('kinematic_coupling'):
        from motiogen.evalkit.k.coupling_extension import apply as extend_coupling
        extension=extend_coupling(modelpath,config['kinematic_coupling'])
    model=osim.Model(str(modelpath));state=model.initSystem();bindings=couplers(model,state);cs=model.getCoordinateSet()
    records={};arrays={};bodies=[model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())]
    for name,change in domain['changes'].items():
        c=cs.get(name);lo,hi=c.getRangeMin(),c.getRangeMax();original=change['range_native_before']
        # 0.1 degree coordinate spacing, including endpoints and original bounds.
        grid=np.unique(np.r_[np.linspace(lo,hi,int(np.ceil(np.rad2deg(hi-lo)*10))+1),np.clip(original,lo,hi)])
        related=[]
        for ji in range(model.getJointSet().getSize()):
            joint=osim.CustomJoint.safeDownCast(model.getJointSet().get(ji))
            if joint is None:continue
            st=joint.getSpatialTransform()
            for axis in range(6):
                ta=st.getTransformAxis(axis);coords=ta.getCoordinateNames()
                names=[ta.get_coordinates(i) for i in range(coords.size())]
                if name in names:
                    related.append((joint.getName()+'/'+str(axis),ta.getFunction(),names,axis<3))
        functions=[];fk=[];coupling_error=[]
        for q in grid:
            c.setValue(state,float(q),False);synchronize(state,bindings);model.realizePosition(state)
            vals=[]
            for _,fun,names,_ in related:vals.append(fun.calcValue(osim.Vector([cs.get(n).getValue(state) for n in names])))
            functions.append(vals)
            transforms=[]
            for body in bodies:
                tf=model.getBodySet().get(body).getTransformInGround(state)
                transforms.append([*[tf.p().get(i) for i in range(3)],*[tf.R().get(i,j) for i in range(3) for j in range(3)]])
            fk.append(transforms)
            coupling_error.append(max([abs(dep.getValue(state)-fun.calcValue(osim.Vector([ic.getValue(state) for ic in independent]))) for dep,independent,fun in bindings],default=0))
        values=np.asarray(functions);pose=np.asarray(fk);rots=pose[:,:,3:].reshape(len(grid),-1,3,3)
        det_error=float(np.max(abs(np.linalg.det(rots)-1)));orthogonality=float(np.max(abs(rots.swapaxes(-1,-2)@rots-np.eye(3))));finite=bool(np.isfinite(values).all() and np.isfinite(pose).all())
        spatial=np.linalg.norm(np.diff(pose[:,:,:3],axis=0),axis=-1)
        deltar=rots[1:]@rots[:-1].swapaxes(-1,-2)
        steps=np.rad2deg(Rotation.from_matrix(deltar.reshape(-1,3,3)).magnitude()).reshape(len(grid)-1,-1)
        fnrows=[]
        for j,(fn,_,_,angular) in enumerate(related):
            unit=180/np.pi if angular else 1000;v=values[:,j]*unit;inside=(grid>=original[0])&(grid<=original[1]);grad=np.gradient(v,np.rad2deg(grid))
            fnrows.append({'function':fn,'units':'deg' if angular else 'mm','original_min_max':[float(v[inside].min()),float(v[inside].max())],
                'expanded_min_max':[float(v.min()),float(v.max())],'max_abs_slope_per_deg':float(abs(grad).max()),'original_max_abs_slope_per_deg':float(abs(grad[inside]).max())})
        # Explicit numeric guard only: per0.1deg coordinate step <=2deg frame
        # rotation and <=1cm body-origin shift, far above observed old-domain steps.
        passed=finite and det_error<1e-8 and orthogonality<1e-8 and max(coupling_error)<1e-8 and steps.max()<2 and spatial.max()<.01
        records[name]={'numerically_passed':bool(passed),'samples':len(grid),'range_deg':np.rad2deg([lo,hi]).tolist(),
            'max_rotation_step_deg':float(steps.max()),'max_body_origin_step_mm':float(spatial.max()*1000),'rotation_determinant_error':det_error,'rotation_orthogonality_error':orthogonality,
            'max_coordinate_coupling_error':max(coupling_error),'functions':fnrows}
        arrays[name+'_grid_rad']=grid;arrays[name+'_functions']=values
        c.setValue(state,c.getDefaultValue(),False);synchronize(state,bindings)
    result={'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
        'source_model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
        'coupling_extension':extension,'bodies_checked':bodies,'domain':domain,'coordinates':records,'passed':all(x['numerically_passed'] for x in records.values()),
        'scope':'Numerical finiteness, continuity and declared coupling checks only; no physiological, muscle or force validation',
        'guards':'0.1degree grid; max frame rotation2deg and body-origin shift1cm perstep, determinant/coupling tolerance1e-8; explicit engineering checks'}
    np.savez_compressed(a.output/'functions.npz',**arrays);(a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('passed',result['passed']);print(json.dumps(records,indent=2))
    if not result['passed']:raise RuntimeError('Coupling audit failed')


if __name__=='__main__':main()
