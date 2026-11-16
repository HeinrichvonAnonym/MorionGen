#!/usr/bin/env python3
"""Aggregate fixed-window K3 ablations by recording and participant; select on development only."""
import argparse,json,hashlib,collections,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import region
VARIANTS=['legacy_raw','legacy_projected','optimized_raw','optimized_projected']


def record_score(row,variant):
    values=row['scores'][variant]['raw']['angles'];groups=collections.defaultdict(list)
    for metric,s in values.items():
        value=s['end_to_end']['p95_abs']
        if value is not None:groups[region(metric)].append(value)
    return float(np.mean([np.mean(v) for v in groups.values()])) if len(groups)==6 else None


def participant_mean(rows,getter):
    values=collections.defaultdict(list)
    for row in rows:
        value=getter(row)
        if value is not None and np.isfinite(value):values[row['group_id']].append(value)
    return float(np.mean([np.mean(v) for v in values.values()])) if values else None


def region_score(row,variant,body_region,stage='end_to_end'):
    values=[s[stage]['p95_abs'] for metric,s in row['scores'][variant]['raw']['angles'].items()
            if region(metric)==body_region and s[stage]['p95_abs'] is not None]
    return float(np.mean(values)) if values else None


def choose(rows):
    scores={v:participant_mean(rows,lambda r:record_score(r,v)) for v in VARIANTS}
    # Predeclared engineering tie margin; prefer simpler unprojected/legacy within 0.1 deg.
    best=min(scores,key=lambda v:float('inf') if scores[v] is None else scores[v]);value=scores[best]
    if value is None:return 'legacy_raw',scores
    acceptable=[v for v in VARIANTS if scores[v] is not None and scores[v]<=value+.1]
    return next(v for v in ['legacy_raw','optimized_raw','legacy_projected','optimized_projected'] if v in acceptable),scores


def build(rows,expected):
    done=[r for r in rows if r['status']=='computed'];result={'expected_exports':expected,'completed':len(done),'failed':len(rows)-len(done),
      'scope':'development-only; original-reference endpoint; complete four-cell cases for policy selection',
      'aggregation':'record-level P95 then participant means; six body regions equally weighted for selection; head-thorax separate',
      'selection_tie_margin_deg':.1,'methods':{},'failures':[{k:r.get(k) for k in ['id','method','status','reason','failures']} for r in rows if r['status']!='computed' or r.get('failures')]}
    for method in sorted({r['method'] for r in rows}):
        available=[r for r in done if r['method']==method];paired=[r for r in available if all(v in r['scores'] for v in VARIANTS)]
        lab=[r for r in paired if r['subset']=='LAB'];selected,scores=choose(lab or paired)
        entry={'available_records':len(available),'paired_records':len(paired),'participant_groups':len({r['group_id'] for r in paired}),
               'selected':selected,'selection_source':'LAB development; industrial subsets independent transfer diagnostics','selection_scores':scores,'variants':{},'folds':[]}
        for held in sorted({r['group_id'] for r in lab}):
            train=[r for r in lab if r['group_id']!=held];test=[r for r in lab if r['group_id']==held]
            variant,train_scores=choose(train)
            entry['folds'].append({'held_group':held,'selected':variant,'training_scores':train_scores,'recordings':len(test),
                'held_original_reference_score':participant_mean(test,lambda r:record_score(r,variant)),
                'held_legacy_score':participant_mean(test,lambda r:record_score(r,'legacy_raw'))})
        for variant in VARIANTS:
            entry['variants'][variant]={'body_pose_score':participant_mean(paired,lambda r:record_score(r,variant)),
                'industrial_pose_score':participant_mean([r for r in paired if r['subset']!='LAB'],lambda r:record_score(r,variant)),
                'angles':{},'positions':{},'coverage':{}}
            output=entry['variants'][variant]
            if not paired:continue
            output['regions']={reg:{stage:participant_mean(paired,lambda r:region_score(r,variant,reg,stage))
                for stage in ['degradation','fit_deviation','end_to_end','projection','fit_after_projection']}
                for reg in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']}
            output['region_precision']={}
            for reg in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']:
                metrics=[metric for metric in paired[0]['scores'][variant]['raw']['angles'] if region(metric)==reg]
                output['region_precision'][reg]={stat:participant_mean(paired,
                    lambda r:float(np.mean([r['scores'][variant]['raw']['angles'][metric]['end_to_end'][stat]
                        for metric in metrics if r['scores'][variant]['raw']['angles'][metric]['end_to_end'][stat] is not None])))
                    for stat in ['median_abs','p95_abs','signed_bias','within_5','within_10','within_20','coverage']}
                output['region_precision'][reg]['metric_frames']={key:sum(
                    r['scores'][variant]['raw']['angles'][metric]['end_to_end'][key]
                    for r in paired for metric in metrics) for key in ['valid','total']}
                output['region_precision'][reg]['scope']='participant-balanced means over explicitly listed geometric angles; frame counts include each metric separately'
                output['region_precision'][reg]['metrics']=metrics
            for metric in paired[0]['scores'][variant]['raw']['angles']:
                output['angles'][metric]={stage:{stat:participant_mean(paired,lambda r:r['scores'][variant]['raw']['angles'][metric][stage][stat]) for stat in ['median_abs','p95_abs','signed_bias','within_5','within_10','within_20','coverage']}
                    for stage in ['degradation','fit_deviation','end_to_end','projection','fit_after_projection','supported_end_to_end']}
            output['directions']={}
            for bone in paired[0]['scores'][variant]['raw']['directions']:
                output['directions'][bone]={stage:{stat:participant_mean(paired,lambda r:r['scores'][variant]['raw']['directions'][bone][stage][stat])
                    for stat in ['median_abs','p95_abs','within_5','within_10','within_20','coverage']}
                    for stage in ['degradation','fit_deviation','end_to_end','projection','fit_after_projection']}
            for joint in ['left_wrist','right_wrist','left_foot','right_foot','pelvis_mid']:
                output['positions'][joint]={space:{stage:participant_mean(paired,lambda r:r['scores'][variant][space]['positions'][joint][stage]['world']['p95_abs']) for stage in ['degradation','fit_deviation','end_to_end','projection','fit_after_projection']}
                                            for space in ['raw','normalized']}
                output['positions'][joint]['normalized_root_relative_end_to_end']=participant_mean(paired,lambda r:r['scores'][variant]['normalized']['positions'][joint]['end_to_end']['root_relative']['p95_abs'])
                output['positions'][joint]['length_component_end_to_end']=participant_mean(paired,lambda r:r['scores'][variant]['length_component']['end_to_end'][joint]['p95_abs'])
            for reg in paired[0]['scores'][variant]['coverage']:
                output['coverage'][reg]={'angle':participant_mean(paired,lambda r:r['scores'][variant]['coverage'][reg]['share']),
                    'registered_angle':participant_mean(paired,lambda r:r['scores'][variant]['registered_angle_coverage'][reg]),
                    'registered_position':participant_mean(paired,lambda r:r['scores'][variant]['position_coverage'][reg])}
            output['contact']={side:{field:participant_mean(paired,lambda r:r['scores'][variant]['contact']['F:'+variant][side][field])
                              for field in ['near_floor_share','below_floor_share','speed_when_near_floor_mps']} for side in ['left','right']}
            output['contact_stages']={stage:{side:{field:participant_mean(paired,lambda r:r['scores'][variant]['contact'][key][side][field])
                for field in ['near_floor_share','below_floor_share','speed_when_near_floor_mps']} for side in ['left','right']}
                for stage,key in [('original','R'),('degraded','D'),('projected','P:'+variant),('fitted','F:'+variant)]}
            output['scoring_frames']={metric:{'valid':sum(r['scores'][variant]['raw']['angles'][metric]['end_to_end']['valid'] for r in paired),
                'total':sum(r['scores'][variant]['raw']['angles'][metric]['end_to_end']['total'] for r in paired)} for metric in output['angles']}
        entry['paired_projection_effect']={strategy:{'pose_score_delta_deg':participant_mean(paired,lambda r:record_score(r,strategy+'_projected')-record_score(r,strategy+'_raw')),
            'per_region_delta_deg':{reg:participant_mean(paired,lambda r:region_score(r,strategy+'_projected',reg)-region_score(r,strategy+'_raw',reg))
                                   for reg in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']}}
            for strategy in ['legacy','optimized']}
        entry['regressions']=[]
        for r in paired:
            delta=record_score(r,selected)-record_score(r,'legacy_raw')
            if delta>1:entry['regressions'].append({'id':r['id'],'body_pose_delta_deg':delta})
        entry['head_thorax']={}
        for part in ['head','thorax','neck']:
            eligible=[r for r in paired if r['head_thorax'][part]['status']=='computed']
            entry['head_thorax'][part]={'records':len(eligible),'total_records':len(paired),
                'p95_deg':participant_mean(eligible,lambda r:r['head_thorax'][part]['end_to_end']['p95_deg']),
                'valid_frames':sum(r['head_thorax'][part]['end_to_end']['valid'] for r in eligible),
                'total_frames':sum(r['frames'] for r in paired),
                'scope':'native K path under explicit controlled convention; model/anatomical assumptions kept'}
        entry['fold_mean_score']=float(np.mean([r['held_original_reference_score'] for r in entry['folds']])) if entry['folds'] else None
        entry['fold_mean_legacy_score']=float(np.mean([r['held_legacy_score'] for r in entry['folds']])) if entry['folds'] else None
        result['methods'][method]=entry
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--partial',action='store_true')
    ap.add_argument('--score-file',default='result_original_lengths.json',choices=['result_original_lengths.json','result.json']);a=ap.parse_args()
    exports=json.loads((a.out/'exports.json').read_text());paths=list((a.out/'fits').glob('*/*/'+a.score_file));rows=[json.loads(p.read_text()) for p in paths]
    if not a.partial and len(rows)!=len(exports):raise ValueError(f'Incomplete: {len(rows)}/{len(exports)}')
    result=build(rows,len(exports));result['partial']=a.partial;result['result_paths']=[str(p) for p in paths];result['score_file']=a.score_file
    dest=ROOT/'config/evalkit_k3_20261116'/('partial_summary.json' if a.partial else 'results_summary.json');dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('records',len(rows),'expected',len(exports))
    for m,r in result['methods'].items():print(m,r['paired_records'],r['selected'],r['selection_scores'])

if __name__=='__main__':main()
