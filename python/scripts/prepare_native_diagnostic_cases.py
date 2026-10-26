"""Select transparent tail examples for inspection, never for estimation."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write

source=ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
analysis=json.loads(source.read_text())
# One declared diagnostic per method; all complete blocks remain in statistics.
metrics={'motionlcm':'hand_position_error','omnicontrol':'jerk_rms',
         'chois':'native_object_mean_penetration_score','ardy':'jerk_rms',
         'dart':'new_goal_final_error_m','primal':'boundary_jerk_rms',
         'closd':'new_goal_final_error_m'}
complete={(b['method'],b['task'],b['seed'],b['clock']) for b in analysis['blocks'] if b['complete']}
entries=[];selection=[]
for method,metric in metrics.items():
    candidates=[r for r in analysis['rows'] if r['method']==method and r['clock']=='paused'
                and (method,r['task'],r['seed'],r['clock']) in complete
                and all(v==1 for v in r['condition'].values()) and metric in r['values']]
    if not candidates:continue
    chosen=min(candidates,key=lambda r:(-r['values'][metric],r['task'],r['seed']))
    case=f"diagnostic_{metric}_task{chosen['task']}_seed{chosen['seed']}"
    rule={'method':method,'case':case,'task':chosen['task'],'seed':chosen['seed'],
          'selection_metric':metric,'selection_value':chosen['values'][metric],
          'unit':chosen['units'].get(metric),'eligible_complete_blocks':len(candidates),
          'analysis_cutoff':analysis['updated_at'],
          'rule':'Maximum diagnostic metric in all-components-on condition, paused clock; ties by task then seed. Show every paired condition. Outcome-selected illustration, not a representative sample or a change to the full-matrix estimator.'}
    selection.append(rule)
    for row in analysis['rows']:
        if (row['method'],row['task'],row['seed'],row['clock'])!=(method,chosen['task'],chosen['seed'],'paused'):continue
        entry={**row,'scene':row['task_spec'],'case':case,'diagnostic_selection':rule}
        if row.get('variant'):entry['motion_path']=row['run']+'/factorial/'+row['variant']+'_canonical.npz'
        entries.append(entry)
out=ROOT/'develop_logs/test_runs/20261024_native_diagnostic_cases'
json_write(out/'entries.json',entries)
json_write(out/'selection.json',{'analysis_cutoff':analysis['updated_at'],'rows':selection})
print(json.dumps({'methods':len(selection),'conditions':len(entries),'selection':selection},ensure_ascii=False))
