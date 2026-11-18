"""Input-contract eligibility; geometry equivalence never uses scores or R."""

POLICIES={'observed_chain_proxy_v1','observed_chain_total_proxy_v1'}


def validate(candidates,requirements):
    if set(requirements)-set(candidates):
        raise ValueError('Requirements name an undeclared candidate')
    for name,rule in requirements.items():
        if not set(rule)<= {'input_rig','geometry_alternative_to'} or rule.get('input_rig') not in ('required','absent'):
            raise ValueError('Candidate eligibility must use declared input fields')
        other=rule.get('geometry_alternative_to')
        if other is None:continue
        if rule['input_rig']!='absent' or other not in candidates or other==name:
            raise ValueError('Geometry alternative needs a distinct declared position candidate')
        a,b=candidates[name],candidates[other]
        if a.get('position_trunk_strategy') not in POLICIES or b.get('position_trunk_strategy') not in POLICIES:
            raise ValueError('Geometry equivalence requires explicit supported proxy policies')
        without_policy=lambda config:{k:v for k,v in config.items() if k!='position_trunk_strategy'}
        if without_policy(a)!=without_policy(b):
            raise ValueError('Geometry alternatives may differ only in fixed trunk proxy policy')
        if requirements.get(other,{}).get('input_rig')!='absent':
            raise ValueError('Base geometry candidate must share the position-input requirement')


def assess(motion,name,candidates,requirements):
    rule=requirements.get(name)
    if not rule:return True,{'status':'applicable','reason':'no_additional_input_requirements'}
    declared=bool(motion.provenance.get('input_rig'))
    record={'requirement':rule,'declared_input_rig_present':declared}
    if declared!=(rule['input_rig']=='required'):
        return False,dict(record,status='inapplicable',reason='input_rig_requirement')
    other=rule.get('geometry_alternative_to')
    if other:
        from .position_geometry import trunk_chain_proxy
        previous,old=trunk_chain_proxy(motion,candidates[other]['position_trunk_strategy'])
        proposed,new=trunk_chain_proxy(motion,candidates[name]['position_trunk_strategy'])
        changed=previous!=proposed
        record.update(base_candidate=other,base_proxy=old,proposed_proxy=new,
                      geometry_target_changed=changed,
                      scope='Exact equality of clip-fixed target or shared model-prior fallback; no error-based exclusion, no R access')
        if not changed:
            return False,dict(record,status='inapplicable',reason='same_fixed_geometry_as_retained_base')
    return True,dict(record,status='applicable',reason='declared_input_requirements_satisfied')
