"""Post-hoc contact illustration; never changes the registered estimator."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write

base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
coverage=next(r for r in analysis['matrix_acceptance'] if r['method']=='chois')
if coverage['accepted_cells']!=120 or coverage['complete_blocks']!=60:
    raise ValueError('Select this illustration only after the full CHOIS matrix')
metric='native_object_mean_contact_f1_score'
effect=next(r for r in analysis['contrasts'] if r['method']=='chois' and r['metric']==metric)
task=min(effect['sensitivity']['task_effects'],key=lambda r:(r['mean_delta'],r['task']))
if task['mean_delta']>=0:raise ValueError('No observed negative scenario mean to illustrate')
candidates=[r for r in analysis['rows'] if r['method']=='chois' and r['task']==task['task']]
seed=min(r['seed'] for r in candidates)
selected=[r for r in candidates if r['seed']==seed]
if len(selected)!=2:raise ValueError('Expected one complete guidance pair')
case=f"contact_counterexample_task{task['task']}_seed{seed}"
rule={'method':'chois','case':case,'task':task['task'],'seed':seed,
    'selection_metric':'scenario_mean_paired_contact_f1_delta','selection_value':task['mean_delta'],
    'unit':'fraction','eligible_complete_blocks':60,'analysis_cutoff':analysis['updated_at'],
    'description':'事后接触反例：完整 12 场景中先选接触 F1 平均配对差最低的场景，再选最小 seed 展示两条件；不是随机代表样本，不改变全部 60 对的统计估计。',
    'rule':'Post-hoc minimum scenario-mean paired F1 effect, ties by task; first seed by numeric order. Not preregistered sample selection for estimation.'}
entries=[{**r,'scene':r['task_spec'],'case':case,'diagnostic_selection':rule} for r in selected]
out=base/'20261024_chois_contact_counterexample'
json_write(out/'entries.json',entries);json_write(out/'selection.json',rule)
print(json.dumps({'selection':rule,'conditions':[{'run':r['run'],'condition':r['condition'],'F1':r['values'][metric]} for r in selected]},ensure_ascii=False))
