"""Join accepted native paired results and posture ranges on identical motion paths."""
import collections
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import json_write,checksum
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
files=[BASE/'posture/analysis.json',BASE/'posture_rotations/analysis.json',BASE/'posture_primal_rotations_v2/analysis.json']
records={}
for path in files:
    for row in json.loads(path.read_text())['rows']:
        if not row['status'].endswith('_computed'):continue
        current=records.setdefault(row['motion_path'],{})
        if 'left' in row.get('ergonomic',{}):
            row['ergonomic']={s+'_'+m:v for s,methods in row['ergonomic'].items() for m,v in methods.items()}
        current.update(row)
groups=collections.defaultdict(list)
for row in records.values():
    for component,value in row['condition'].items():
        if value not in [0,1] or isinstance(value,str):continue
        others={k:v for k,v in row['condition'].items() if k!=component}
        key=(row['method'],str(row['task']),row['seed'],row['clock'],component,json.dumps(others,sort_keys=True))
        groups[key].append(row)
paired=[];incomplete=[]
for key,members in groups.items():
    off=[r for r in members if r['condition'][key[4]]==0]
    on=[r for r in members if r['condition'][key[4]]==1]
    if len(off)!=1 or len(on)!=1:
        incomplete.append(dict(key=key,off_count=len(off),on_count=len(on)));continue
    off,on=off[0],on[0]
    metrics={}
    for metric in sorted(set(off['source_metrics'])|set(on['source_metrics'])):
        a,b=off['source_metrics'].get(metric),on['source_metrics'].get(metric)
        if isinstance(a,(float,int)) and isinstance(b,(float,int)):
            metrics[metric]={'off':a,'on':b,'on_minus_off':b-a}
    ergonomics={}
    for side in ['left_RULA','right_RULA','left_REBA','right_REBA']:
        a,b=off['ergonomic'][side],on['ergonomic'][side]
        value={'off':a,'on':b}
        if all('mean_lower_bound' in x for x in [a,b]):
            value['conservative_mean_score_difference_range']=[b['mean_lower_bound']-a['mean_upper_bound'],b['mean_upper_bound']-a['mean_lower_bound']]
        ergonomics[side]=value
    paired.append(dict(method=key[0],task=key[1],seed=key[2],clock=key[3],component=key[4],
        other_conditions=json.loads(key[5]),off_motion=off['motion_path'],on_motion=on['motion_path'],
        metrics=metrics,ergonomics=ergonomics,
        failed_task_conditions=[label for label,r in [('off',off),('on',on)] if r['source_metrics'].get('position_success_rate')==0]))
json_write(BASE/'joint_correction_tradeoffs.json',dict(pairs=paired,incomplete_pairs=incomplete,
    accepted_motion_count=len(records),source_files={str(p):checksum(p) for p in files},
    scope=['Every accepted motion retained, including zero task success. Absent/non-numeric metrics are missing, not zero.',
           'One component changes at a time conditional on every other registered component; no marginal pooling across interactions.',
           'Clock conditions are separate strata, not independent duplicate samples.',
           'Score difference intervals are conservative envelopes over unknown inputs, not confidence intervals or complete validated totals.',
           'Input/seed matching inherits the frozen accepted native experiment protocol. This join alone is not an independent rerun or native-interface audit.']))
print(json.dumps({'accepted_motions':len(records),'paired_contrasts':len(paired),'incomplete_groups':len(incomplete)}))
