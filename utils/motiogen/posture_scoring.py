"""Score observed continuous angle series without inventing task conditions."""
import numpy as np
from .ergonomics import score

def score_angle_series(angles, fps):
    arrays=dict(angles)
    count=len(next(iter(angles.values())))
    summaries={}
    for side in ['left','right']:
        for method in ['RULA','REBA']:
            bounds=[];levels=[];components={};missing=set();complete=0
            for frame in range(count):
                inputs={}
                for name in ['elbow_flexion_deg','upper_arm_flexion_deg','wrist_flexion_deg']:
                    value=angles.get(side+'_'+name)
                    if value is not None and np.isfinite(value[frame]):inputs[name]=float(value[frame])
                for name in ['neck_flexion_deg','trunk_flexion_deg']:
                    value=angles.get(name)
                    if value is not None and np.isfinite(value[frame]):inputs[name]=float(value[frame])
                if method=='REBA':
                    value=angles.get(side+'_knee_flexion_deg')
                    if value is not None and np.isfinite(value[frame]):inputs['knee_flexion_deg']=float(value[frame])
                result=score(inputs,method)
                missing.update(result['missing_inputs']);complete+=not result['missing_inputs']
                bounds.append(result['score_range'])
                levels.append([min(result['possible_action_levels']),max(result['possible_action_levels'])])
                for name,values in result['components'].items():components.setdefault(name,[]).append([min(values),max(values)])
            prefix=side+'_'+method
            bounds=np.asarray(bounds,dtype=np.int8);levels=np.asarray(levels,dtype=np.int8)
            arrays[prefix+'_bounds']=bounds;arrays[prefix+'_action_level_bounds']=levels
            for name,values in components.items():arrays[prefix+'_component_'+name]=np.asarray(values,dtype=np.int8)
            summaries[prefix]=dict(frames=count,complete_total_frames=int(complete),
                mean_lower_bound=float(bounds[:,0].mean()),mean_upper_bound=float(bounds[:,1].mean()),
                score_range=[int(bounds.min()),int(bounds.max())],missing_inputs=sorted(missing),
                definite_high_action_seconds=float(np.sum(levels[:,0]>=3)/fps),
                possible_high_action_seconds=float(np.sum(levels[:,1]>=3)/fps),
                scope='Unknown task and qualitative adjustment inputs enumerated; bounds are not confidence intervals or human-validated totals.')
    arrays['fps']=fps
    return arrays,summaries
