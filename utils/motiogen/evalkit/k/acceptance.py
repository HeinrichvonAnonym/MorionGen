"""Development acceptance of saved scores; never used inside the fitter."""


def compare(baseline,candidate):
    regressions=[];checked=[];unsupported=[]
    def check(path,a,b,limit,field='p95',unit='deg'):
        if a.get('n',0)==0:
            unsupported.append(path);return
        ac=a['n']/a['denominator'];bc=b['n']/b['denominator'] if b['denominator'] else 0
        if bc<ac-.01-1e-12:
            regressions.append({'metric':path,'kind':'coverage','baseline':ac,'candidate':bc,'limit':.01})
        if b.get(field) is None:return
        row={'metric':path,'kind':field,'baseline':a[field],'candidate':b[field],'limit':limit,'unit':unit}
        checked.append(row)
        if b[field]>a[field]+limit+1e-9:regressions.append(row)
    for comparison in ('F-D','F-R'):
        # Keep full-window checks. Source-event tolerance views are reported
        # separately; K-created jumps do not acquire a blanket exemption.
        a=baseline['comparisons'][comparison]['full'];b=candidate['comparisons'][comparison]['full']
        for group in ('directions_deg','joint_angles_deg'):
            for name,old in a[group].items():check(f'{comparison}/{group}/{name}',old,b[group][name],1.)
        for region,nodes in a['positions_cm'].items():
            for name,old in nodes.items():
                for geometry in ('world','pelvis_relative','common_length_world','common_length_pelvis_relative'):
                    check(f'{comparison}/{region}/{name}/{geometry}',old[geometry],b['positions_cm'][region][name][geometry],1.,unit='cm')
        for name,old in a['temporal'].items():
            if 'output_step_deg' not in old:continue
            new=b['temporal'][name]
            check(f'{comparison}/{name}/step',old['output_step_deg'],new['output_step_deg'],5.,field='max')
            check(f'{comparison}/{name}/direction_rate',old['direction_rate_vector_error_deg_s'],new['direction_rate_vector_error_deg_s'],10.,unit='deg/s')
    return {'status':'regression_blocked' if regressions else 'no_regression_detected',
            'regressions':regressions,'checked':checked,'unsupported':unsupported,
            'scope':'Development comparison; passing regression gates alone does not establish target attainment or improvement'}
