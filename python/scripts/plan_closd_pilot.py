"""Pair the native planner context switch while retaining the same PHC/PhysX."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import json_write
cfg=load_config(); data=Path(cfg['data_root']); assets=data/'outputs/assets/closd-official'
jobs=[]
for source in ['simulation','prediction']:
    jobs.append({'id':'closd_native_'+source+'_seed20','method':'closd','seed':20,
        'scene':{'name':'native_physics_root_goal_change','duration_seconds':20,'update_seconds':5},
        'method_spec':{'repo':'baselines/CLoSD','environment':'motiogen-closd'},
        'command':[str(data/'envs/motiogen-closd/bin/python'),str(ROOT/'python/scripts/closd_native.py'),
                   '--run-dir','{run_dir}','--context-source',source,'--frames','600','--seed','20'],
        'required_outputs':['{run_dir}/motion.npz','{run_dir}/metrics.json','{run_dir}/intervention.json','{run_dir}/native_physics.npz'],
        'require_protocol':False,'timeout_seconds':1800,'estimated_seconds':300})
identity=[assets/'source_receipt.json',assets/'checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth',
          assets/'checkpoints/dip/DiP_multi-target_10steps_context20_predict40/model000300000.pt',
          data/'outputs/assets/closd-distilbert/model.safetensors']
identity.extend((assets/'data/smpl').glob('SMPL_*.pkl'))
destination=ROOT/'develop_logs/test_runs/20261024_closd_pilot/plan.json'
if destination.exists():raise ValueError('Pilot plan exists; preserve it and create an explicit revision')
json_write(destination,{'devices':[0,1],'deadline':json.loads((ROOT/'config/survey_revision.json').read_text())['experiment_deadline_at'],
    'report_reserve_seconds':14400,'jobs':jobs,'identity_files':list(map(str,identity))})
print(destination)
