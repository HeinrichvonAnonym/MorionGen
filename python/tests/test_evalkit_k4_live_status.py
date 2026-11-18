"""Pure-stdlib presentation tests; no fitting or numerical worker required."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('k4_live',ROOT/'python/scripts/evalkit_k4_live_report.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)


class LiveStatus(unittest.TestCase):
    def test_waiting_has_full_denominator_and_distinct_prerequisites(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as folder:
            run=Path(folder)
            (run/'protocol.json').write_text(json.dumps(dict(workers=5,records=[dict(window='000',method='a'),dict(window='000',method='b')])))
            (run/'progress.json').write_text(json.dumps(dict(state='waiting_for_registered_runs',stage=None,waiting_for=['/data/pre1/completion.json','/data/pre1','/data/pre2'])))
            result=live.Results(run).snapshot()
            self.assertEqual(result['expected'],2);self.assertEqual(result['pending'],2)
            self.assertEqual(result['waiting_on'],['pre1','pre2'])
            self.assertEqual(result['recorded_state'],'waiting_for_registered_runs')
            self.assertIsNone(result['latest_result_at'])

    def test_metric_failure_is_completed_not_execution_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as folder:
            run=Path(folder)
            (run/'protocol.json').write_text(json.dumps(dict(workers=5,records=[dict(window='000',method='a'),dict(window='000',method='b')])))
            (run/'progress.json').write_text(json.dumps(dict(state='validation_stage_failed',stage='registered_transfer')))
            task=run/'tasks/000_a';task.mkdir(parents=True)
            (task/'result.json').write_text(json.dumps(dict(status='completed',passed=False)))
            before=(task/'result.json').read_bytes()
            result=live.Results(run).snapshot()
            self.assertEqual((result['evaluated'],result['execution_failed'],result['pending']),(1,0,1))
            self.assertEqual(result['stage'],'registered_transfer')
            self.assertIsNotNone(result['latest_result_at'])
            self.assertEqual(before,(task/'result.json').read_bytes())


if __name__=='__main__':unittest.main()
