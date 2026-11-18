"""Separate observed-input fidelity from unobserved inserted-path diagnostics."""

def quality_groups(quality):
    groups={'observed':[],'inserted_path':[],'unsupported':[]}
    for name,value in quality['metrics'].items():
        group='unsupported' if not value['expected'] else 'inserted_path' if name.startswith('interpolation_') else 'observed'
        groups[group].append({'metric':name,**value})
    inserted=any(x['metric'].startswith('interpolation_direction/') for x in groups['inserted_path'])
    return dict(groups=groups,has_inserted_observations=False,has_inserted_samples=inserted,
                observed_failures=[x for x in groups['observed'] if not x['passed']],
                path_failures=[x for x in groups['inserted_path'] if not x['passed']] if inserted else [],
                scope='Inserted-path constraints are interpolation assumptions, not independent motion observations.')
