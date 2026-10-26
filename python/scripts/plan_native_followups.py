"""Coverage-first follow-up blocks and CHOIS matrix; preserve original receipts."""
import copy
import json
import random
import sys
from pathlib import Path
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,environment_root
from motiogen.runtime import checksum,json_write

cfg=load_config(); data=Path(cfg['data_root'])
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
primal=json.loads((base/'primal_plan.json').read_text())
dart=json.loads((base/'dart_plan.json').read_text())
chois=json.loads((base/'chois_plan.json').read_text())
jobs=[]
initial=copy.deepcopy(primal['jobs'][0]); initial['id']='primal_exact600_stand_seed20'
initial['scene']={'name':'native_continuation','initial_history':0,'initial_seed':9000,'frames':600}
initial['command'][initial['command'].index('--initial-history')+1]='0'
initial['estimated_seconds']=240
jobs.append(initial)
for old in dart['jobs']:
    job=copy.deepcopy(old); job['id']=job['id']+'_realtime'
    job['scene']['clock']='realtime';job['command']+=['--clock','realtime'];job['estimated_seconds']=60
    jobs.append(job)
processed=data/'outputs/assets/chois/processed_data'
source=processed/'cano_test_diffusion_manip_window_120_joints24.p'
windows=joblib.load(source)
# Mirror native dataset's text and first-full-window filters exactly (no object split).
filtered=[v for v in windows.values() if (processed/'omomo_text_anno_json_data'/(v['seq_name']+'.json')).exists()]
filtered=[v for v in filtered if len(v['motion'])>=120 and v['start_t_idx']==0]
tasks=[]; objects=set()
for index,entry in enumerate(filtered):
    obj=entry['seq_name'].split('_')[1]
    if obj in objects: continue
    objects.add(obj);tasks.append({'native_window_index':index,'sequence':entry['seq_name'],'object':obj})
    if len(tasks)==12: break
if len(tasks)!=12: raise ValueError('Expected 12 distinct native object tasks')
reused=[]
for task_id,task in enumerate(tasks):
    for seed in range(20,25):
        block=[]
        for condition,old in enumerate(chois['jobs']):
            if task_id==0 and seed==20:
                receipt=json.loads((base/'chois_queue_retry1/state.json').read_text())['jobs'][old['id']]
                if receipt['status']!='success': raise ValueError('CHOIS pilot is not complete')
                reused.append({'task':task_id,'seed':seed,'condition':condition,'run':receipt['run_dir']})
                continue
            job=copy.deepcopy(old); job.update(id=f'chois_task{task_id}_seed{seed}_guidance{condition}',seed=seed,task_id=task_id)
            job['scene'].update(native_window_index=task['native_window_index'],native_sequence=task['sequence'])
            job['estimated_seconds']=180
            block.append(job)
        random.Random(task_id*1000+seed).shuffle(block);jobs.extend(block)
destination=ROOT/'develop_logs/test_runs/20261024_native_followups/plan.json'
if destination.exists(): raise ValueError('Refusing to overwrite fixed follow-up plan')
json_write(destination,{'devices':[0,1],'deadline':json.loads((ROOT/'config/survey_revision.json').read_text())['experiment_deadline_at'],
    'report_reserve_seconds':14400,'jobs':jobs,'tasks':tasks,'reused':reused,
    'identity_files':sorted(set(primal.get('identity_files',[])+dart.get('identity_files',[])+chois.get('identity_files',[])+[str(source)])),
    'selection':'First eligible full native window per object, native index order, first 12 objects; fixed before outcomes',
    'source_windows_sha256':checksum(source)})
print(json.dumps({'plan':str(destination),'jobs':len(jobs),'tasks':tasks,'reused':len(reused)}))
