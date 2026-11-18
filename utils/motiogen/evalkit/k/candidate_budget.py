"""Optional input-only early acceptance; independent reference grading is separate."""
import math


def validate(policy):
    if policy is None:return
    if set(policy)!={'version','margin'} or policy['version']!='first_input_margin_v1':
        raise ValueError('Unknown candidate budget policy')
    if not math.isfinite(policy['margin']) or not 0<policy['margin']<1:
        raise ValueError('Candidate margin must be strictly between zero and one')


def accepts(quality, policy):
    validate(policy)
    if policy is None or not quality.get('export_path_diagnostic') or not quality.get('passed'):
        return False
    metrics=[v for v in quality['metrics'].values() if v['expected']]
    if not metrics:return False
    return all(v['missing']==0 and v['present']==v['expected'] and v['value'] is not None
               and math.isfinite(v['value']) and v['value']>=0
               and math.isfinite(v['threshold']) and v['threshold']>0
               and v['value']<=policy['margin']*v['threshold'] for v in metrics)
