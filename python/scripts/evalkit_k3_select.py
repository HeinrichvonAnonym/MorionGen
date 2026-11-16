#!/usr/bin/env python3
"""Materialize the preregistered per-format development selection after all jobs finish."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    folder=ROOT/'config/evalkit_k3_20261116';path=folder/'results_summary.json';r=json.loads(path.read_text())
    if r.get('partial') or r['completed']+r['failed']!=r['expected_exports']:
        raise ValueError('Selection requires the completed registered experiment')
    config=json.loads((folder/'experiment_config.json').read_text());policies={}
    for method,result in r['methods'].items():
        v=result['selected']
        if result['paired_records']<1:raise ValueError('No paired evidence for '+method)
        policies[method]={'strategy':'direction_refinement' if v.startswith('optimized') else 'legacy','projection':v.endswith('projected')}
    config['version']='k-3-corrected-bvh-development-selected'
    config['fitting']['per_method']=policies
    config['selection']={'scope':'development selection, no new final holdout','rule':'config/evalkit_k3_20261116/selection_rule.json',
                         'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'experiment_commit':json.loads((folder/'experiment_freeze.json').read_text())['commit'],
                         'default_compatibility':'K2 remains default; choose this config explicitly'}
    (ROOT/'config/evalkit_k_v3.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(policies,indent=2))

if __name__=='__main__':main()
