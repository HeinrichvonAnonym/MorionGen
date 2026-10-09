#!/usr/bin/env python
"""Build a real-motion E proxy reference from the complete KIT-ML test split."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.datasets import kit21_motion
from motiogen.metrics import evaluate
from motiogen.runtime import json_write
from mld.data.humanml.scripts.motion_process import recover_from_ric

METRICS=['jerk_rms','trunk_inclination','trunk_risk_fraction','left_upper_arm_elevation',
         'left_upper_arm_risk_fraction','right_upper_arm_elevation','right_upper_arm_risk_fraction',
         'left_elbow_rom','right_elbow_rom']


def percentile(value, reference):
    arr=np.asarray(reference,float)
    # Midrank keeps a common minimum such as risk_fraction=0 near the middle
    # of its tie block instead of misleadingly placing it at the block's top.
    return float(100*(np.sum(arr<value)+0.5*np.sum(arr==value))/len(arr))


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(argv); data=Path(args.dataset); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    ids=[x.strip() for x in (data/'test.txt').read_text().splitlines() if x.strip()]
    rows=[]; values={k:[] for k in METRICS}; failures=[]
    scene={'name':'kit_ml_reference','kind':'reference','duration_seconds':1,'prompt':'','constraints':[]}
    for item in ids:
        try:
            features=np.load(data/'new_joint_vecs'/f'{item}.npy'); direct=np.load(data/'new_joints'/f'{item}.npy')
            recovered=recover_from_ric(torch.from_numpy(features).float(),21).numpy()
            if recovered.shape!=direct.shape or not np.allclose(recovered,direct,atol=1e-5):
                raise ValueError('official RIC recovery disagrees with published new_joints')
            motion=kit21_motion(recovered,12.5); joints=motion['joints']
            result=evaluate(motion,scene); row={'id':item,'frames':len(joints)}
            for name in METRICS:
                metric=result['metrics'].get(name,{}); value=metric.get('value') if metric.get('status')=='measured' else None
                row[name]=value
                if value is not None: values[name].append(value)
            rows.append(row)
        except Exception as exc:
            failures.append({'id':item,'reason':f'{type(exc).__name__}: {exc}'})
    with (out/'per_case.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={'schema_version':1,'status':'success' if not failures else 'partial','scope':'KIT-ML test split real-motion E proxy reference; not FID/R-Precision and not a model ranking',
             'fps':12.5,'units':'metres after official KIT RIC recovery','expected':830,'measured':len(rows),'failures':failures,
             'metrics':{name:{'n':len(vals),'p05':float(np.percentile(vals,5)),'p50':float(np.percentile(vals,50)),
                              'p95':float(np.percentile(vals,95))} for name,vals in values.items() if vals}}
    cases=json.loads((ROOT/'config/abeh_report_cases.json').read_text())['cases']; placements=[]
    for case in cases:
        metrics_file=ROOT/case['run_dir']/'metrics.json'
        if not metrics_file.is_file(): continue
        metrics=json.loads(metrics_file.read_text()).get('metrics',{})
        item={'case_id':case['id'],'baseline':case['baseline']}
        for name,reference in values.items():
            metric=metrics.get(name,{})
            if metric.get('status')=='measured': item[name]={'value':metric['value'],'kit_percentile':percentile(metric['value'],reference)}
        placements.append(item)
    summary['baseline_case_percentiles']=placements; json_write(out/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'measured':len(rows),'failures':len(failures),'output':str(out)},ensure_ascii=False))
    return 0 if not failures else 2


if __name__=='__main__': raise SystemExit(main())
