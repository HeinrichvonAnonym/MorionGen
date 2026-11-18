import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.experiment_recovery import prepare,sha
class RecoveryTests(unittest.TestCase):
 def test_identity_rejection_precedes_archive_and_restart_preserves_partial(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as t:
   p=Path(t);out=p/'out';out.mkdir();base=p/'base';code=p/'code';config=p/'config';config.mkdir();model=p/'model';model.write_text('asset');source=p/'D';source.write_text('motion')
   identity={'code_commit':'frozen','config_sha256':{},'new_holdout_uses':0,'workers':1};protocol={'conditions':['candidate']};panel={'records':[{'window':'000','method':'format'}]}
   for n,v in [('identity',identity),('protocol',protocol),('panel',panel)]: (out/(n+'.json')).write_text(json.dumps(v))
   common=base/'common_v1/000/format';common.mkdir(parents=True);(common/'receipt.json').write_text(json.dumps({'key':{'inputs':{str(source):sha(source)},'model_sha256':sha(model)}}))
   task=out/'tasks/candidate/000_format';task.mkdir(parents=True);(task/'fit.log').write_text('partial evidence')
   with patch('motiogen.evalkit.experiment_recovery.subprocess.run'),patch.dict(sys.modules,{'biomech':SimpleNamespace(MODEL=model)}):
    with self.assertRaises(ValueError):prepare(out,code,config,base,{**identity,'code_commit':'changed'},protocol,panel)
    self.assertTrue(task.exists());self.assertFalse((out/'attempt_history').exists())
    reused,restart=prepare(out,code,config,base,identity,protocol,panel)
   self.assertEqual(reused,[]);self.assertEqual(len(restart),1);self.assertFalse(task.exists())
   self.assertEqual(next((out/'attempt_history').rglob('fit.log')).read_text(),'partial evidence')
   self.assertEqual(json.loads((out/'identity.json').read_text()),identity)
if __name__=='__main__':unittest.main()
