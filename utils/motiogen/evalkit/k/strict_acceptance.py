"""Per-window engineering targets with input-defined observability, never fit selection."""
import math


def assess(score, baseline, policy):
    if not score.get('fixed_observation_support'):raise ValueError('Strict acceptance requires common frozen R/D support')
    required=[];diagnostic=[];temporal=[];coverage=[]
    stratum=policy['acceptance_stratum']
    if stratum!='continuous':raise ValueError('Frozen continuous/source-jump split required')
    source=score['comparisons']['D-R'][stratum];fitted=score['comparisons']['F-R'][stratum]
    def target(path,observed,value,threshold,bucket=required):
        expected=observed.get('n',0);actual=value.get('n',0) if value else 0
        p95=value.get('p95') if value else None
        finite=p95 is not None and math.isfinite(p95)
        state=('unobservable' if expected==0 else 'coverage_failed' if actual<expected
               else 'target_failed' if not finite or p95>threshold+1e-9 else 'passed')
        bucket.append({'metric':path,'status':state,'expected_observed':expected,'scored':actual,
                       'denominator':observed.get('denominator',0),'p95':p95,'target':threshold})
    for group in ('directions_deg','joint_angles_deg'):
        for name,observed in source[group].items():
            target(group+'/'+name,observed,fitted.get(group,{}).get(name),policy['engineering_angle_p95_deg'])
    for region in ('head_neck','shoulders','hands','knees','feet'):
        for name,observed in source['positions_cm'][region].items():
            values=fitted.get('positions_cm',{}).get(region,{}).get(name,{})
            target('positions_cm/'+name+'/world',observed['world'],values.get('world'),policy['engineering_world_position_p95_cm'])
            for geometry in ('pelvis_relative','common_length_world','common_length_pelvis_relative'):
                target('positions_cm/'+name+'/'+geometry,observed[geometry],values.get(geometry),policy['engineering_world_position_p95_cm'],diagnostic)
    rotations=score.get('head_neck_rotation_comparisons')
    if rotations:
        observed=rotations['D-R'][stratum];value=rotations['F-R'][stratum]
    else:
        observed=score.get('native_rotation_readout') or {'n':0,'denominator':score['clock']['frames']}
        value=None
    target('head_relative_thorax_deg',observed,value,policy['engineering_angle_p95_deg'])
    def check_coverage(observed,output,path=''):
        if isinstance(observed,dict) and 'n' in observed and 'denominator' in observed:
            coverage.append({'metric':path,'expected_observed':observed['n'],'scored':output.get('n',0),
                             'status':'passed' if output.get('n',0)==observed['n'] else 'failed'})
        elif isinstance(observed,dict):
            for key,value in observed.items():
                check_coverage(value,output.get(key,{}) if isinstance(output,dict) else {},path+'/'+key)
    for group in ('directions_deg','joint_angles_deg','positions_cm'):
        check_coverage(score['comparisons']['D-R']['full'][group],score['comparisons']['F-R']['full'][group],group)
    if rotations:check_coverage(rotations['D-R']['full'],rotations['F-R']['full'],'head_relative_thorax_deg')
    # New head diagnostics have no old model output; use the observed D trace
    # as their temporal control. Existing body diagnostics retain the old rule.
    stratum=policy['temporal_stratum'].split(';')[0]
    if stratum!='continuous':raise ValueError('Explicit frozen continuous stratum required')
    current=score['comparisons']['F-R'][stratum]['temporal']
    old=baseline['comparisons']['F-R'][stratum]['temporal'] if baseline else {}
    data=score['comparisons']['D-R'][stratum]['temporal']
    for name,trace in current.items():
        if 'output_step_deg' not in trace:continue
        control=old.get(name,{})
        if not control.get('output_step_deg',{}).get('n'):control=data.get(name,{})
        for field,stat,limit in [('output_step_deg','max',policy['temporal_max_step_increase_deg']),
                               ('direction_rate_vector_error_deg_s','p95',policy['temporal_rate_error_p95_increase_deg_s'])]:
            a=control.get(field,{});b=trace.get(field,{})
            if not a.get('n'):continue
            expected=data.get(name,{}).get(field,{}).get('n',0)
            av,bv=a.get(stat),b.get(stat)
            passed=b.get('n',0)>=expected and bv is not None and av is not None and bv<=av+limit+1e-9
            temporal.append({'metric':name+'/'+field,'status':'passed' if passed else 'failed',
                'baseline':av,'candidate':bv,'increase_limit':limit,'expected_observed':expected,'scored':b.get('n',0)})
    head_time=score.get('head_neck_rotation_temporal')
    if head_time:
        output=head_time['F-R']['continuous'];observed=head_time['D-R']['continuous']
        old_head=(baseline or {}).get('head_neck_rotation_temporal',{}).get('F-R',{}).get('continuous',{})
        for field,stat,limit in [('output_step_deg','max',policy['temporal_max_step_increase_deg']),
                               ('angular_velocity_error_deg_s','p95',policy['temporal_rate_error_p95_increase_deg_s'])]:
            control=old_head.get(field,{})
            if not control.get('n'):control=observed[field]
            if not control.get('n'):continue
            value=output[field];av,bv=control.get(stat),value.get(stat)
            ok=value['n']==observed[field]['n'] and av is not None and bv is not None and bv<=av+limit+1e-9
            temporal.append({'metric':'head_relative_thorax/'+field,'status':'passed' if ok else 'failed',
                             'baseline':av,'candidate':bv,'increase_limit':limit})
    failures=[r for r in required if r['status'] not in ('passed','unobservable')]
    return {'version':policy['version'],'status':'passed' if not failures and all(r['status']=='passed' for r in temporal+coverage) else 'failed',
            'required':required,'full_window_coverage':coverage,'acceptance_stratum':stratum,'diagnostic_geometries':diagnostic,'temporal':temporal,
            'failed_targets':len(failures),'unobservable_metrics':sum(r['status']=='unobservable' for r in required),
            'scope':'Absolute original-R engineering targets; observability from D-R, not candidate coverage; no automatic promotion'}
