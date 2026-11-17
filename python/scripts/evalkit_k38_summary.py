"""Finalize root-stage evidence and explicitly retain adverse endpoint changes."""
from evalkit_k38_root import *

def main():
 frozen=read(DEST/'freeze.json')
 assert all(digest(ROOT/p)==h for p,h in frozen['protected'].items())
 assert all(digest(p)==h for p,h in frozen['references'].items())
 assert all(digest(p)==h for p,h in frozen['inputs'].items())
 records=[read(p) for p in read(DEST/'index.json')]
 runtime_root=BASE/'evalkit_k38_runtime_20261117';runtime=[read(p) for p in read(runtime_root/'index.json')]
 changes=[]
 for r in records:
  old=r['scores']['old'];new=r['scores']['fit_anchor_only']
  for name in ['pelvis_mid','shoulder_mid','left_foot','right_foot','left_wrist','right_wrist']:
   oldpos=old['raw']['positions'][name]['end_to_end']['world']['p95_abs'];newpos=new['raw']['positions'][name]['end_to_end']['world']['p95_abs']
   changes.append({'window':r['row']['window'],'method':r['row']['method'],'landmark':name,'old_m':oldpos,'new_m':newpos,'increase_m':newpos-oldpos})
  for name in old['raw']['angles']:
   assert abs(old['raw']['angles'][name]['end_to_end']['p95_abs']-new['raw']['angles'][name]['end_to_end']['p95_abs'])<1e-6
 fixed=next(r for r in records if r['row']['window']=='074');fresh=next(r for r in runtime if r['kind']=='corrected_input')
 a=np.load(Path(fixed['work'])/'both_anchors.npz')['F:output'];b=np.load(Path(fresh['output'])/'aligned.npz')['F:output']
 point_difference=float(np.linalg.norm(a-b,axis=-1).max())
 verification={'protected_unchanged':len(frozen['protected']),'references_unchanged':len(frozen['references']),'frozen_input_files_unchanged':len(frozen['inputs']),'new_holdout_runs':0,'tests_passed':84,'subtests_passed':16,'fresh_vs_fixed_pose_max_point_difference_m':point_difference,'root_scope':'observed midpoint, geometric only; no new rotational information; no force qualification'}
 assert '84 passed, 16 subtests passed' in (ROOT/'develop_logs/test_runs/20261117_k38/tests.log').read_text()
 for r in runtime:
  assert r['cache_replay_equal'] and r['receipt']['evidence']['implementation']==implementation_key()
 config=ROOT/'config/evalkit_k38_20261117'
 def compact_scores(scores):
  return {variant:{'positions':{space:{name:{stage:{kind:values[kind]['p95_abs'] for kind in ['world','root_relative']} for stage,values in metrics.items() if stage in ['degradation','fit_deviation','end_to_end']} for name,metrics in score[space]['positions'].items()} for space in ['raw','normalized']},
      'angle_p95':{name:{stage:values['p95_abs'] for stage,values in metrics.items() if stage in ['degradation','fit_deviation','end_to_end']} for name,metrics in score['raw']['angles'].items()},
      'coverage':score['coverage'],'position_coverage':score['position_coverage'],'root_velocity':score['root_velocity']} for variant,score in scores.items()}
 compact=[]
 for r in records:
  p=Path(r['work'])/'result.json';compact.append({'row':r['row'],'result':str(p),'result_sha256':digest(p),'scores':compact_scores(r['scores']),'root_receipt':r['root_receipt'],'extra':r['extra']})
 runtime_compact=[]
 for r in runtime:
  p=Path(r['output'])/'result.json';runtime_compact.append({'kind':r['kind'],'source':r['source'],'result':str(p),'result_sha256':digest(p),'scores':compact_scores(r['scores']),'cache_replay_equal':r['cache_replay_equal']})
 atomic(config/'acceptance.json',{'fixed_pose_records':compact,'fresh_runtime':runtime_compact,'same_input_endpoint_changes':changes,'verification':verification,'contact_plane_note':'These experiment scores use one original-reference foot p05 plane per window for all cells. It enters scoring only, not root alignment. Generic contact scope labels inherited from scoring.py are not plane provenance.'})
 atomic(config/'freeze.json',{'fixed_pose':frozen,'runtime':read(runtime_root/'freeze.json')});atomic(config/'verification.json',verification)
 print(verification)
 print('same old-input endpoint increases >1cm',[(r['window'],r['landmark'],round(r['increase_m']*100,2)) for r in changes if r['increase_m']>.01])
 print('074 both', {n:round(fresh['scores']['aligned']['raw']['positions'][n]['end_to_end']['world']['p95_abs']*100,4) for n in ['pelvis_mid','left_foot','right_foot']})
if __name__=='__main__':main()
