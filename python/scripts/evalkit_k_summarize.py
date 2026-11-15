#!/usr/bin/env python3
"""Record/person-macro K summaries and pre-registered development folds; never tune on holdout."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def metric_summary(rows,get):
    records=defaultdict(list);groups={};total=valid=0
    for row in rows:
        metric=get(row)
        if not metric:continue
        records[row['recording_id']].append(metric);groups[row['recording_id']]=row['group_id']
        total+=metric.get('total',0);valid+=metric.get('valid',0)
    def record_mean(values,key):
        v=[m[key] for m in values if m.get(key) is not None]
        return float(np.mean(v)) if v else None
    keys=['median_deg','p95_deg','within_5_deg','within_10_deg','within_20_deg']
    by_record={key:[record_mean(v,key) for v in records.values()] for key in keys}
    result={'recordings':len(records),'groups':len(set(groups.values())),
            'valid_frames':valid,'total_frames':total,'frame_coverage':valid/total if total else None}
    for key,values in by_record.items():
        values=[v for v in values if v is not None]
        result['record_macro_'+key]=float(np.mean(values)) if values else None
        if key=='p95_deg':
            result['record_p95_median_deg']=float(np.median(values)) if values else None
            result['record_p95_max_deg']=float(max(values)) if values else None
    by_person=defaultdict(list)
    for rec,metrics in records.items():
        m=record_mean(metrics,'within_10_deg')
        if m is not None:by_person[groups[rec]].append(m)
    result['person_macro_within_10_deg']=float(np.mean([np.mean(v) for v in by_person.values()])) if by_person else None
    bias=[m['signed_rotvec_bias_deg'] for values in records.values() for m in values if m.get('signed_rotvec_bias_deg') is not None]
    result['segment_macro_rotvec_bias_deg']=np.mean(bias,0).tolist() if bias else None
    result['aggregation']='Han segments averaged within original recording; then records equally weighted. P95 values are averaged per record, not pooled percentiles.'
    return result


def gate_choice(rows,region):
    choices=[]
    for gate in ('0.005','0.01','0.02'):
        s=metric_summary(rows,lambda r:r.get('regions',{}).get(region,{}).get('residual_gates',{}).get(gate))
        choices.append((gate,s))
    passing=[x for x in choices if x[1]['person_macro_within_10_deg'] is not None and x[1]['person_macro_within_10_deg']>=.95]
    # A declared engineering trade-off: largest supported coverage at 95% within 10°.
    selected=max(passing,key=lambda x:x[1]['frame_coverage'] or 0)[0] if passing else None
    return {'selected_gate_m':selected,'candidates':dict(choices),'criterion':'person-macro within 10 degrees >= 0.95, then maximum frame coverage; development engineering choice'}


def summarize(rows,partition):
    sections={}
    for dataset in ('andy','han','collaborate'):
        selected=[r for r in rows if r['dataset']==dataset]
        if not selected:continue
        methods=sorted({m for r in selected for m in r.get('profiles',{}).get('methods',{})})
        section={'trials':len(selected),'recordings':len({r['recording_id'] for r in selected}),
                 'groups':len({r['group_id'] for r in selected}),'statuses':dict(Counter(r['status'] for r in selected)),
                 'execution_failures':[{'id':r['id'],'reason':r.get('reason')} for r in selected if r['status']!='computed'],
                 'profile_window_statuses':dict(Counter(r.get('profiles',{}).get('status','failed') for r in selected)),
                 'frames':sum(r.get('frames',0) for r in selected),
                 'position_precision_invalid_trials':[r['id'] for r in selected if r.get('position_precision_invalid_frames',0)],
                 'position_precision_invalid_frames':sum(r.get('position_precision_invalid_frames',0) for r in selected),
                 'reference':metric_summary(selected,lambda r:r.get('neck_template_comparison',r.get('position_thorax_proxy'))),
                 'gaps':{str(g):metric_summary(selected,lambda r,g=g:r.get('gap_recovery',{}).get(str(g))) for g in (.1,.5,1.,2.)},'methods':{}}
        for m in methods:
            rr=lambda r:r.get('profiles',{}).get('methods',{}).get(m,{})
            section['methods'][m]={'statuses':dict(Counter(rr(r).get('status','no_window') for r in selected)),
                                  'runtime_calibration':dict(Counter(rr(r).get('runtime_model_calibration','unavailable') for r in selected))}
            for key in ('retarget_and_temporal','codec','end_to_end','thorax_end_to_end'):
                section['methods'][m][key]=metric_summary(selected,lambda r,key=key:rr(r).get(key))
            section['methods'][m]['body']={key:metric_summary(selected,lambda r,key=key:rr(r).get('body_end_to_end',{}).get(key))
                                          for key in ('trunk_incline','left_knee_flexion','right_knee_flexion','left_elbow_flexion','right_elbow_flexion')}
        sections[dataset]=section
    result={'partition':partition,'datasets':sections}
    if partition=='development':
        folds=json.loads((ROOT/'config/gold_k_20261115/split.json').read_text())['development_folds']
        result['folds']=[]
        for fold in folds:
            train=[r for r in rows if r['group_id'] in fold['training_groups']]
            val=[r for r in rows if r['group_id'] in fold['validation_groups']]
            output={'id':fold['id'],'training_records':len({r['recording_id'] for r in train}),
                    'validation_records':len({r['recording_id'] for r in val}),'validation_groups':fold['validation_groups']}
            if val and val[0]['dataset']!='collaborate':
                output['regions']={}
                for region in ('head','thorax'):
                    choice=gate_choice(train,region);gate=choice['selected_gate_m']
                    output['regions'][region]={**choice,'validation':metric_summary(val,lambda r:r.get('regions',{}).get(region,{}).get('residual_gates',{}).get(gate))}
            else:
                output['fixed_mapping_validation']=metric_summary(val,lambda r:r.get('position_thorax_proxy'))
                output['selection']='native hierarchy and collar mapping fixed from development inspection; no fitted statistical parameters'
            result['folds'].append(output)
        result['selected_reference_gates']={d:{region:gate_choice([r for r in rows if r['dataset']==d],region) for region in ('head','thorax')} for d in ('andy','han')}
    return result


def summarize_gaps(rows):
    result={}
    for dataset in ('andy','han','collaborate'):
        selected=[r for r in rows if r['dataset']==dataset]
        if not selected:continue
        result[dataset]={'trials':len(selected),'statuses':dict(Counter(r['status'] for r in selected)),'regions':{}}
        for region in ('head','thorax','head_and_thorax'):
            result[dataset]['regions'][region]={str(seconds):metric_summary(selected,lambda r:r.get('cases',{}).get(region,{}).get(str(seconds),{}).get('neck'))
                                                for seconds in (.1,.5,1.,2.)}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gap-input',type=Path)
    a=ap.parse_args();receipt=json.loads((a.input/'receipt.json').read_text())
    rows=[json.loads(l) for l in (a.input/'records.jsonl').read_text().splitlines()]
    if len(rows)!=receipt['records'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incomplete or duplicate record set')
    result=summarize(rows,receipt['partition']);result['source']=str(a.input.resolve());result['implementation']=receipt['implementation']
    if a.gap_input:
        gaps=[json.loads(l) for l in (a.gap_input/'records.jsonl').read_text().splitlines()]
        if len(gaps)!=len(rows) or {r['id'] for r in gaps}!={r['id'] for r in rows}:raise ValueError('Gap and format record sets differ')
        result['registered_gap_experiment']=summarize_gaps(gaps)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
