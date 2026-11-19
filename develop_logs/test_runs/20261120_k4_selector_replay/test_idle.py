import ast,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
root=Path('/home/heinrich/MorionGen');spec=importlib.util.spec_from_file_location('replay',root/'python/scripts/evalkit_k4_selector_replay.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117/selector_replay_20261120_v1/code/evalkit_k4_selector_replay.py')
class Tests(unittest.TestCase):
 def test_unchanged_ledger_never_reads_json_or_hashes(self):
  with tempfile.TemporaryDirectory(dir=root/'develop_logs/test_runs/20261120_k4_selector_replay') as t:
   p=Path(t)/'ledger.json';p.write_text(json.dumps({'rows':[{'window':'001','method':'x','fully_evaluated':False}]}));w=m.LedgerWatch(p)
   w.poll()
   with patch.object(m.json,'load',side_effect=AssertionError('unexpected full parse')),patch.object(m,'sha',side_effect=AssertionError('unexpected hash')):
    for _ in range(100):self.assertFalse(w.poll()[1])
   self.assertEqual(w.reads,1);self.assertEqual(w.ready,set())
 def test_atomic_replace_same_size_and_mtime_detected(self):
  with tempfile.TemporaryDirectory(dir=root/'develop_logs/test_runs/20261120_k4_selector_replay') as t:
   p=Path(t)/'ledger.json';p.write_text(json.dumps({'rows':[{'window':'001','method':'x','fully_evaluated':True}]}));w=m.LedgerWatch(p);w.poll();st=p.stat()
   q=Path(t)/'new';q.write_text(p.read_text().replace('001','002'));os.utime(q,ns=(st.st_atime_ns,st.st_mtime_ns));q.replace(p)
   self.assertTrue(w.poll()[1]);self.assertEqual(w.ready,{'002_x'});self.assertEqual(w.reads,2)
 def test_scientific_functions_exactly_unchanged(self):
  def f(p):return {x.name:ast.dump(x) for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef)}
  a,b=f(old),f(Path(m.__file__))
  for n in ('read','sha','write','worker','prepare','audit','allocation'):self.assertEqual(a[n],b[n],n)
if __name__=='__main__':unittest.main()
