"""Pure-stdlib presentation tests; no fitting or numerical worker required."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('k4_live',ROOT/'python/scripts/evalkit_k4_live_report.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)


class LiveStatus(unittest.TestCase):
    def make_ablation(self):
        return dict(expected_routes=656,applicable_cells=100,independently_evaluated_cells=80,
                    candidate_passed_cells=60,candidate_metric_failed_cells=20,execution_failed_cells=1,pending_cells=19,
                    fully_evaluated_routes=5,all_candidates_passed_routes=2,selector_evaluated_routes=10,
                    selector_passed_routes=9,selector_missed_passing_routes=1,selector_comparison_complete_routes=5,
                    per_candidate=[dict(name='candidate <A>',applicable=10,evaluated=8,passed=6,metric_failed=2,execution_failed=0,pending=2)],
                    rows=[dict(window='000',method='a',selected='candidate <A>')])

    def test_missing_ablation_is_unknown_not_zero_failures(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as folder:
            run=Path(folder);(run/'protocol.json').write_text(json.dumps(dict(workers=6,records=[])))
            data=live.Results(run,run/'missing').ablation()
            self.assertFalse(data['available']);self.assertNotIn('candidate_metric_failed_cells',data)
            document=live.ablation_document(data)
            self.assertIn('未知',document);self.assertNotIn('指标未通过 0',document)

    def test_candidate_completion_and_selector_success_stay_separate(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as folder:
            run=Path(folder);(run/'protocol.json').write_text(json.dumps(dict(workers=6,records=[])))
            progress=run/'progress.json';progress.write_text(json.dumps(self.make_ablation()));before=progress.read_bytes()
            data=live.Results(run,run).ablation();document=live.ablation_document(data)
            self.assertIn('独立验收完成 5/656',document)
            self.assertIn('所有候选均通过 2/656',document)
            self.assertIn('选择器输出验收：通过 9/656',document)
            self.assertIn('指标未通过 20',document);self.assertIn('执行失败 1',document)
            self.assertIn('6/10',document);self.assertIn('candidate &lt;A&gt;',document)
            self.assertEqual(before,progress.read_bytes())

    def test_partial_ablation_does_not_invent_complete_statistics(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as folder:
            run=Path(folder);(run/'protocol.json').write_text(json.dumps(dict(workers=6,records=[])))
            (run/'progress.json').write_text('{"expected_routes":656,"fully_evaluated_routes":0}')
            data=live.Results(run,run).ablation()
            self.assertFalse(data['available']);self.assertNotIn('candidate_passed_cells',data)

    def test_readonly_injection_labels_frozen_selector_table(self):
        frozen='<html><h1>Frozen</h1><p>通过 12/656</p><table><th>通过／总数</th></table></html>'
        result=live.enhance_report(frozen,dict(available=False,reason='未知'),'// test')
        self.assertIn('选择器输出验收：通过 12/656',result)
        self.assertIn('选择器通过／总数',result)
        self.assertIn('id="k4-ablation"',result)
        self.assertNotIn('选择器输出验收',frozen)

    def test_route_candidate_table_keeps_unknown_and_inapplicable_distinct(self):
        data={**self.make_ablation(),'available':True}
        data['rows']=[dict(window='000',method='a',fully_evaluated=False,selector={},candidates={
            'waiting':dict(status='pending',passed=None),
            'unsupported':dict(status='inapplicable',reason='input contract'),
            'bad':dict(status='completed',passed=False,failures=['pose'])})]
        document=live.ablation_document(data)
        self.assertIn('候选验收未完成',document)
        self.assertIn('<td>waiting</td><td>待验</td><td>未知</td>',document)
        self.assertIn('不计适用分母',document)
        self.assertIn('<td>bad</td><td>已独立验收</td><td>未通过</td>',document)

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
