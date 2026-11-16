#!/usr/bin/env python3
"""Report local regressions and near-identity projection sensitivity after selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python/scripts'))
from evalkit_k3_report import record_score,region_score
from evalkit_k3_constraints import inspect_tail
REGIONS=['pelvis','torso','left_arm','right_arm','left_leg','right_leg']


def main():
    cfg=ROOT/'config/evalkit_k3_20261116';summary=json.loads((cfg/'results_summary.json').read_text())
    if summary['partial']:raise ValueError('Requires the complete registered experiment')
    rows=[json.loads(Path(p).read_text()) for p in summary['result_paths']]
    result={'scope':'post-selection diagnostics; fixed masks and policies; region-level regressions are retained even when the overall score improves',
            'regression_threshold_deg':1.,'methods':{}}
    for method,entry in summary['methods'].items():
        pool=[r for r in rows if r['method']==method and r['status']=='computed'];variant=entry['selected'];records=[]
        for r in pool:
            regions={reg:{'baseline':region_score(r,'legacy_raw',reg),'selected':region_score(r,variant,reg)} for reg in REGIONS}
            for v in regions.values():v['delta']=v['selected']-v['baseline']
            bones=r['scores']['optimized_projected']['raw']['positions']
            record={'id':r['id'],'window_key':r['window_key'],'group_id':r['group_id'],
                'baseline':record_score(r,'legacy_raw'),'selected':record_score(r,variant),'regions':regions,
                'projection':{'pose_score_delta_deg':record_score(r,'optimized_projected')-record_score(r,'optimized_raw'),
                    'endpoint_displacement_p95_m':max(bones[j]['projection']['world']['p95_abs'] for j in ['left_wrist','right_wrist','left_foot','right_foot'])}}
            record['delta']=record['selected']-record['baseline']
            records.append(record)
        ranked=sorted(records,key=lambda r:r['selected'],reverse=True)
        regional={reg:{'regressed_records':[r['id'] for r in records if r['regions'][reg]['delta']>1.],
            'largest_increase':max(records,key=lambda r:r['regions'][reg]['delta']),
            'highest_remaining_error':max(records,key=lambda r:r['regions'][reg]['selected'])} for reg in REGIONS}
        sensitivity=sorted(records,key=lambda r:abs(r['projection']['pose_score_delta_deg']),reverse=True)[:5]
        for r in sensitivity:
            folder=Path(next(x['arrays'] for x in pool if x['id']==r['id'])).parent
            r['projection']['initial_arm_candidate']={}
            for setting in ['raw','projected']:
                path=folder/('base_'+setting)/'receipt.json';receipt=json.loads(path.read_text())
                r['projection']['initial_arm_candidate'][setting]=receipt.get('arm_selection',{}).get('selected','legacy_fixed_strategy')
        result['methods'][method]={'records':records,'worst_remaining':ranked[:5],
            'overall_increase_count':sum(r['delta']>0 for r in records),'overall_increase_over_1deg_count':sum(r['delta']>1 for r in records),
            'regions':regional,'largest_projection_sensitivity':sensitivity}
        focus=max(regional,key=lambda reg:regional[reg]['highest_remaining_error']['regions'][reg]['selected'])
        case=regional[focus]['highest_remaining_error']
        result['methods'][method]['worst_region_constraints']=inspect_tail(next(r for r in pool if r['id']==case['id']),variant,focus)
    (cfg/'tail_diagnostics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for m,e in result['methods'].items():
        print(m,'region increases >1deg',{k:len(v['regressed_records']) for k,v in e['regions'].items()},
            'worst remaining',e['worst_remaining'][0]['id'],round(e['worst_remaining'][0]['selected'],2))


if __name__=='__main__':main()
