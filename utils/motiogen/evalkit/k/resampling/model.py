"""SO(3) interpolation of eligible model joints, with explicit domain fallback.

Locked joints remain fixed. Single-axis joints retain their equivalent scalar
geodesic when native increments are below pi. Coupled/partial-DOF joints stay
on their declared coordinate manifold. No reference or scoring data are read.
"""
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from ..config import digest


def audit_model(path):
    root=ET.parse(path);dependent=set();coupled=set()
    for constraint in root.findall('.//CoordinateCouplerConstraint'):
        dependent.update(constraint.findtext('dependent_coordinate_name','').split())
        coupled.update(constraint.findtext('independent_coordinate_names','').split())
    joints={}
    for joint in root.findall('.//JointSet/objects/*'):
        coords={c.get('name'):{'range_rad':np.fromstring(c.findtext('range',''),sep=' ').tolist(),
                 'locked':c.findtext('locked','false')=='true','clamped':c.findtext('clamped','false')=='true',
                 'dependent':c.get('name') in dependent} for c in joint.findall('./coordinates/Coordinate')}
        row={'type':joint.tag,'coordinates':coords};names=list(coords)
        row['classification']='constrained_or_unsupported_manifold'
        if coords and all(c['locked'] for c in coords.values()):row['classification']='locked_fixed'
        elif any(n in dependent or n in coupled for n in coords):row['classification']='coordinate_coupled_manifold'
        elif joint.tag=='PinJoint' and len(coords)==1:row['classification']='one_axis_geodesic_equivalent'
        elif joint.tag=='CustomJoint':
            order='';rot_names=[];signs=[];eligible=True
            for i in range(3):
                axis=joint.find(f"./SpatialTransform/TransformAxis[@name='rotation{i+1}']")
                if axis is None:eligible=False;break
                vector=np.fromstring(axis.findtext('axis',''),sep=' ')
                coefficients=np.fromstring(axis.findtext('./LinearFunction/coefficients',''),sep=' ')
                name=axis.findtext('coordinates','').strip()
                if vector.shape!=(3,) or coefficients.shape!=(2,) or abs(coefficients[0])!=1 or coefficients[1]!=0 or name not in coords:
                    eligible=False;break
                index=int(np.argmax(abs(vector)))
                if not np.array_equal(abs(vector),np.eye(3)[index]) or coords[name]['locked']:
                    eligible=False;break
                order+='XYZ'[index];rot_names.append(name);signs.append(float(coefficients[0]*vector[index]))
            # Rotation-dependent translation would couple SO(3) to another path.
            translations=[a.findtext('coordinates','').split() for a in joint.findall('./SpatialTransform/TransformAxis') if a.get('name','').startswith('translation')]
            if eligible and len(set(order))==3 and len(set(rot_names))==3 and not any(set(n)&set(rot_names) for n in translations):
                row.update(classification='three_axis_so3',order=order,rotation_coordinates=rot_names,signs=signs)
        joints[joint.get('name')]=row
    return {'version':'whole-body-rotation-audit-1','model_sha256':digest(path),'joints':joints,
            'root_translation':'independent scalar interpolation; no SE(3) claim',
            'coupling_policy':'preserve coordinate functions and synchronize dependent coordinates before FK'}


def _lift(raw,guess,lower,upper):
    """Nearest equivalent 2pi lifts in the declared coordinate box."""
    lo=np.ceil((lower-raw-1e-10)/(2*np.pi));hi=np.floor((upper-raw+1e-10)/(2*np.pi))
    winding=np.clip(np.round((guess-raw)/(2*np.pi)),lo,hi)
    values=raw+2*np.pi*winding
    valid=np.all(lo<=hi,axis=-1)
    return np.minimum(np.maximum(values,lower),upper),valid


def _represent(rotations,order,signs,guess,lower,upper):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',UserWarning)
        canonical=rotations.as_euler(order)
    other=canonical.copy();other[:,0]+=np.pi;other[:,1]=np.pi-other[:,1];other[:,2]+=np.pi
    candidates=[];costs=[]
    for angles in (canonical,other):
        q,valid=_lift(angles/signs,guess,lower,upper)
        costs.append(np.where(valid,np.sum((q-guess)**2,axis=1),np.inf));candidates.append(q)
    choice=np.argmin(costs,axis=0);out=np.stack(candidates)[choice,np.arange(len(guess))]
    valid=np.isfinite(np.min(costs,axis=0))
    # At a gimbal pole, first/last coordinates have a continuum of equivalent
    # pairs. Find a bounded pair rather than mistaking the canonical gauge for
    # a physical range violation. Only exact poles use this extra family.
    for i in np.flatnonzero(abs(np.cos(canonical[:,1]))<1e-10):
        phase,b,_=canonical[i];s=1.
        trial=Rotation.from_euler(order,[phase-1,b,1])
        if (rotations[i].inv()*trial).magnitude()>1e-8:s=-1.
        angle_lo=np.minimum(lower*signs,upper*signs);angle_hi=np.maximum(lower*signs,upper*signs)
        desired=guess[i]*signs
        sum_lo=angle_lo[0]+min(s*angle_lo[2],s*angle_hi[2]);sum_hi=angle_hi[0]+max(s*angle_lo[2],s*angle_hi[2])
        kmin=int(np.ceil((sum_lo-phase-1e-10)/(2*np.pi)));kmax=int(np.floor((sum_hi-phase+1e-10)/(2*np.pi)))
        if kmin>kmax:continue
        center=int(np.clip(round((desired[0]+s*desired[2]-phase)/(2*np.pi)),kmin,kmax))
        for k in set([max(kmin,center-1),center,min(kmax,center+1)]):
            total=phase+2*np.pi*k
            cmin=max(angle_lo[2],min((total-angle_lo[0])/s,(total-angle_hi[0])/s))
            cmax=min(angle_hi[2],max((total-angle_lo[0])/s,(total-angle_hi[0])/s))
            if cmin>cmax:continue
            c=np.clip((s*(total-desired[0])+desired[2])/2,cmin,cmax)
            q,ok=_lift(np.array([[total-s*c,b,c]])/signs,guess[i:i+1],lower,upper)
            if ok[0] and (rotations[i].inv()*Rotation.from_euler(order,q[0]*signs)).magnitude()<1e-8:
                cost=np.sum((q[0]-guess[i])**2)
                if not valid[i] or cost<np.sum((out[i]-guess[i])**2):out[i]=q[0];valid[i]=True
    error=(rotations.inv()*Rotation.from_euler(order,out*signs)).magnitude()
    valid &= error<1e-8
    return out,valid,len(caught)


def _matrix(frame,state):
    transform=frame.getTransformInGround(state);r=transform.R()
    return np.array([[r.get(i,j) for j in range(3)] for i in range(3)])


def interpolate_rotations(model_path,labels,coordinates,times,new_times,interpolated):
    import opensim as osim
    import biomech
    from ..constraints import couplers,synchronize
    labels=list(labels);coordinates=np.asarray(coordinates);times=np.asarray(times);new_times=np.asarray(new_times)
    if (len(times)<2 or np.any(np.diff(times)<=0) or np.any(np.diff(new_times)<=0)
        or not np.isfinite(times).all() or not np.isfinite(new_times).all()
        or new_times[0]<times[0]-1e-10 or new_times[-1]>times[-1]+1e-10
        or coordinates.shape!=(len(times),len(labels)) or not np.isfinite(coordinates).all()):
        raise ValueError('Finite ordered native and bounded output clocks/coordinates required')
    new_times=np.clip(new_times,times[0],times[-1]);result=np.array(interpolated,copy=True)
    if result.shape!=(len(new_times),len(labels)) or not np.isfinite(result).all():raise ValueError('Output coordinate layout/finite-value mismatch')
    audit=audit_model(model_path);model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    eligible={n:r for n,r in audit['joints'].items() if r['classification']=='three_axis_so3'}
    observed={n:[] for n in eligible}
    # Verify every native local rotation against actual OpenSim frame transforms.
    for q in coordinates:
        biomech._set(model,state,labels,q);synchronize(state,bindings);model.realizePosition(state)
        for name in eligible:
            joint=model.getJointSet().get(name)
            observed[name].append(_matrix(joint.getParentFrame(),state).T@_matrix(joint.getChildFrame(),state))
    left=np.clip(np.searchsorted(times,new_times,side='right')-1,0,len(times)-2)
    counts={};intervals={};knot_errors={}
    for name,row in audit['joints'].items():
        if row['classification']=='one_axis_geodesic_equivalent':
            coordinate=next(iter(row['coordinates']));delta=np.diff(coordinates[:,labels.index(coordinate)])
            row['max_native_step_deg']=float(abs(delta).max());row['short_arc_equivalence_verified']=bool(np.all(abs(delta)<180-1e-8))
            row['interpolation']='scalar on declared one-axis subgroup; equivalent to SO(3) short arc for reported increments'
        if name not in eligible:continue
        cols=[labels.index(n) for n in row['rotation_coordinates']];signs=np.asarray(row['signs']);order=row['order']
        rotations=Rotation.from_euler(order,np.deg2rad(coordinates[:,cols])*signs)
        verification=float(abs(np.asarray(observed[name])-rotations.as_matrix()).max())
        if verification>1e-8:raise ValueError('OpenSim/declared rotation mismatch: '+name)
        bounds=np.asarray([row['coordinates'][n]['range_rad'] for n in row['rotation_coordinates']]);lo,hi=bounds[:,0],bounds[:,1]
        # Also evaluate exact native knots and interval midpoints. An infeasible
        # sampled geodesic makes the entire interval use the bounded scalar path.
        check=np.unique(np.r_[new_times,times,(times[:-1]+times[1:])/2])
        target=Slerp(times,rotations)(check)
        guess=np.stack([np.interp(check,times,coordinates[:,j]) for j in cols],axis=1)*np.pi/180
        q,valid,warnings_count=_represent(target,order,signs,guess,lo,hi)
        native_index=np.searchsorted(check,times)
        if not valid[native_index].all():raise ValueError('Native rotation lacks an equivalent in-domain representation: '+name)
        indices=np.clip(np.searchsorted(times,check,side='right')-1,0,len(times)-2)
        bad=set(indices[~valid].tolist())
        step=np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude())
        ambiguous=set(np.flatnonzero(step>180-1e-7).tolist());bad |= ambiguous
        chosen=q[np.searchsorted(check,new_times)]*180/np.pi
        fallback=np.isin(left,list(bad));result[:,cols]=np.where(fallback[:,None],result[:,cols],chosen)
        # Native coordinate charts can differ, but native rotations must remain.
        knot_error=float((rotations.inv()*Rotation.from_euler(order,q[native_index]*signs)).magnitude().max())
        knot_errors[name]=knot_error
        records=[{'interval':i,'start_s':float(times[i]),'end_s':float(times[i+1]),
                  'reason':'half_turn_path_ambiguous' if i in ambiguous else 'sampled_geodesic_has_no_in_domain_Euler_representative'} for i in sorted(bad)]
        intervals[name]=records;counts[name]={'native_intervals':len(times)-1,'so3_intervals':len(times)-1-len(bad),
            'fallback_intervals':len(bad),'output_frames':len(new_times),'fallback_output_frames':int(fallback.sum()),
            'native_fk_matrix_error':verification,'native_rotation_preservation_rad':knot_error,'gimbal_warning_count':warnings_count}
        # Fallback does not clip to a geodesic or remove frames from evaluation.
        if np.any(np.deg2rad(result[:,cols])<lo-1e-8) or np.any(np.deg2rad(result[:,cols])>hi+1e-8):
            raise ValueError('Output violates declared coordinate domain: '+name)
    return result,{'version':'whole-body-so3-1','audit':audit,'counts':counts,'fallback_intervals':intervals,
        'native_rotation_preservation_rad':knot_errors,
        'scope':'Same native fitted poses and fixed model; SO(3) eligible joints; explicit whole-interval bounded fallback without dropping coverage',
        'assumption':'Shortest local joint-rotation path; does not recover unobserved high-frequency motion or guarantee angular-velocity continuity',
        'head_neck':'Separate attached-chain path: observed rotations already use mask-aware SLERP; position-only twist stays unobserved'}
