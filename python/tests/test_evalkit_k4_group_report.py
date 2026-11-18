import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'utils/motiogen/evalkit/reporting/aggregation.py';s=importlib.util.spec_from_file_location('aggregation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Groups(unittest.TestCase):
    def record(self,w,method,status='completed',observable=True,p95=1):
        metric={'metric':'head','status':'passed' if observable else 'unobservable','p95':p95 if observable else None,'target':5,'expected_observed':2 if observable else 0,'scored':2 if observable else 0}
        acceptance={v:{c:{'required':[metric],'temporal':[],'full_window_coverage':[]} for c in ('D_clock','R_clock')} for v in ('native','export')}
        return dict(window=w,method=method,group_id='registered/A',status=status,passed=True if status=='completed' else None,acceptance=acceptance)
    def test_unknown_pending_and_execution_failure_remain_distinct(self):
        r=[self.record('0','a'),self.record('1','a',observable=False),self.record('2','a','pending'),self.record('3','a','failed')]
        g=m.summarize(r)['by_format']['a'];q=g['views']['native']['D_clock']['required']['head']
        self.assertEqual(g['expected_paths'],4);self.assertEqual(g['execution_failed_paths'],1);self.assertEqual(g['pending_paths'],1)
        self.assertEqual(q['status_counts'],{'passed':1,'unobservable':1});self.assertEqual(q['ungraded_paths'],2);self.assertEqual(q['window_p95_count'],1)
    def test_paired_formats_remain_separate_and_window_percentile_is_labelled(self):
        r=[self.record('0','a',p95=1),self.record('1','a',p95=3),self.record('0','b',p95=4)]
        out=m.summarize(r);self.assertEqual(len(out['by_participant_format']),2)
        a=out['by_participant_format']['registered/A | a'];q=a['views']['export']['R_clock']['required']['head']
        self.assertEqual(a['original_windows'],2);self.assertEqual(q['mean_window_p95'],2);self.assertEqual(q['worst_window_p95'],3);self.assertNotIn('pooled_p95',q)
    def test_duplicate_metric_and_changed_threshold_rejected(self):
        a=self.record('0','a');b=self.record('1','a');b['acceptance']['native']['D_clock']['required'][0]['target']=6
        with self.assertRaisesRegex(ValueError,'Inconsistent target'):m.summarize([a,b])
        a=self.record('0','a');a['acceptance']['native']['D_clock']['required']*=2
        with self.assertRaisesRegex(ValueError,'Repeated metric'):m.summarize([a])
if __name__=='__main__':unittest.main()
