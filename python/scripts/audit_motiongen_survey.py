"""Post-run integrity receipt; originals remain unchanged."""
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'develop_logs/test_runs/20261024_feedback_replication_v4'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
summary=json.loads((RUN/'summary.json').read_text())
assert summary['n_runs']==288 and len(summary['prefix_checks'])==144
receipts=[]
for row in summary['rows']:
    directory=ROOT/row['source']
    files=['motion.npz','native_pose.npz','metrics.json','trace.json']
    for file in files:assert (directory/file).is_file()
    xyz=np.load(directory/'motion.npz')['joints']
    assert xyz.shape==(450,22,3) and np.isfinite(xyz).all()
    trace=json.loads((directory/'trace.json').read_text())
    assert trace['protocol_version']==4 and 0 <= trace['hold_fraction'] <= 1
    receipts.append({'case':row['source'],'files':{name:sha(directory/name) for name in files}})
code=['utils/motiogen/feedback.py','utils/motiogen/feedback_benchmark.py','utils/motiogen/dart_feedback.py',
      'python/scripts/dart_feedback_smoke.py','python/scripts/primal_feedback_smoke.py',
      'python/scripts/summarize_feedback_replication.py','python/scripts/build_motiongen_survey_data.py',
      'config/motiongen_survey_methods.json']
source_receipt=json.loads((ROOT/'develop_logs/test_runs/20261024_feedback_reproducibility.json').read_text())
receipt={'created_at':datetime.now(timezone.utc).isoformat(),'runs':288,'episode_frames':450,
         'files':receipts,'code_sha256':{p:sha(ROOT/p) for p in code},
         'summary_sha256':sha(RUN/'summary.json'),
         'manifests':{m:json.loads((RUN/m/'manifest.json').read_text()) for m in ['dart','primal']},
         'assets':source_receipt['assets'],'environment_pip_freeze':source_receipt['environment_pip_freeze'],
         'asset_receipt_source':'develop_logs/test_runs/20261024_feedback_reproducibility.json',
         'scope':'Post-run episode hashes; existing asset hashes copied from prior checked receipt, not publisher signatures.',
         'commands':['CUDA_VISIBLE_DEVICES=0 <motiogen-dart-python> python/scripts/dart_feedback_smoke.py --suite',
                     'CUDA_VISIBLE_DEVICES=1 <motiogen-dart-python> python/scripts/primal_feedback_smoke.py --suite',
                     '/home/heinrich/anaconda3/bin/python python/scripts/summarize_feedback_replication.py',
                     '/home/heinrich/anaconda3/bin/python python/scripts/build_motiongen_survey_data.py --complete'],
         'project_git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)}
(RUN/'integrity_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps({'runs':288,'hashed_outputs':len(receipts)*4,'receipt':str(RUN/'integrity_receipt.json')}))
