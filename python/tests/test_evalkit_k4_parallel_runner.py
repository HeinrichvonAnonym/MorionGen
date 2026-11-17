import importlib.util,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('k4_parallel_runner',ROOT/'python/scripts/evalkit_k4_optimize.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class ParallelRunnerTests(unittest.TestCase):
    def test_three_child_limit_and_runtime_reference_isolation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as tmp:
            root=Path(tmp);run=root/'run';config=root/'config';config.mkdir();(run/'jobs').mkdir(parents=True)
            rows=[{'window':str(i),'method':'known'} for i in range(6)]
            (config/'protocol.json').write_text(json.dumps({'conditions':['candidate'],'strict_scoring':True}))
            (config/'sentinels.json').write_text(json.dumps({'records':rows}));(config/'candidate.json').write_text('{}')
            key={'inputs':{'D':'hash'},'model_sha256':'model'}
            for r in rows:
                (run/'jobs'/f"{r['window']}_known.json").write_text(json.dumps({'input':'D','method':'known','mass_kg':75,'floor_m':0,'reference':'forbidden'}))
                p=run/'common_v1'/r['window']/'known';p.mkdir(parents=True);(p/'receipt.json').write_text(json.dumps({'key':key}))
            lock=threading.Lock();active=0;peak=0;payloads=[]
            def child(args,**kwargs):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                try:
                    time.sleep(.01)
                    if Path(args[1]).name=='evalkit_k4_run.py':
                        job=json.loads(Path(args[args.index('--job')+1]).read_text());payloads.append(job)
                        folder=Path(job['output']);folder.mkdir(parents=True);(folder/'receipt.json').write_text(json.dumps({'key':key}))
                    return SimpleNamespace(returncode=0)
                finally:
                    with lock:active-=1
            output=root/'output';argv=['runner','--run',str(run),'--code-root',str(ROOT),'--config-root',str(config),'--output',str(output),'--workers','3']
            with patch.object(sys,'argv',argv),patch.object(runner.subprocess,'run',side_effect=child),patch.object(runner.subprocess,'check_output',return_value=b'frozen'),patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
                runner.main()
            self.assertLessEqual(peak,3);self.assertGreaterEqual(peak,2);self.assertEqual(len(payloads),6)
            self.assertTrue(all(set(p)=={'input','method','mass_kg','floor_m','output','fit_config'} for p in payloads))
            done=json.loads((output/'completion.json').read_text());self.assertEqual(len(done['results']),6)
            self.assertTrue(all(r['status']=='completed' for r in done['results']))

if __name__=='__main__':unittest.main()
