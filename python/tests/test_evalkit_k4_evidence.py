"""Clock/output labels and unknown observations must survive report compaction."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('evidence',Path(__file__).resolve().parents[2]/'utils/motiogen/evalkit/reporting/evidence.py')
evidence=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(evidence)

class ClockEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.task=self.root/'tasks/000_m';self.task.mkdir(parents=True)
        self.record={'window':'000','method':'m','group_id':'subjectA'}
        self.dump(self.task/'result.json',dict(self.record,status='completed',passed=True))
        self.dump(self.root/'selection.json',{'selected':'a','budget':{'early_accepted':'a'}})
        scores={};self.expected={}
        for view,start in [('native',10),('export',20)]:
            def block(value):return {'clock':{'frames':value},'comparisons':{'F-R':{'continuous':{'n':value,'p95':None,'signed_bias':-.3}}},'head_neck_rotation_comparisons':{'F-R':{'n':0,'p95':None}}}
            score=block(start);score['original_clock']=block(start+1)
            path=self.root/(view+'.json');self.dump(path,score);scores[view]=str(path)
            self.expected[view]={'D_clock':start,'R_clock':start+1}
        self.summary=dict(self.record,passed=True,selected='a',score_files=scores,candidate_selection_file=str(self.root/'selection.json'),
            acceptance={v:{c:{'status':'passed'} for c in ['D_clock','R_clock']} for v in ['native','export']},export_audit={'passed':True},failed_metrics=[])
        self.dump(self.task/'summary.json',self.summary)
    def dump(self,p,v):p.write_text(json.dumps(v))
    def test_four_distinct_clocks_and_unknowns_are_exact(self):
        v=evidence.read_record(self.root,self.record)
        for view,clocks in self.expected.items():
            for clock,n in clocks.items():
                b=v['views'][view][clock];self.assertEqual(b['clock']['frames'],n)
                stat=b['comparisons']['F-R']['continuous'];self.assertEqual(stat['n'],n);self.assertIsNone(stat['p95']);self.assertEqual(stat['signed_bias'],-.3)
        self.assertEqual(len(v['source_sha256']),5)
    def test_mismatched_selection_rejected(self):
        self.dump(self.root/'selection.json',{'selected':'another'})
        with self.assertRaisesRegex(ValueError,'choice differs'):evidence.read_record(self.root,self.record)
    def test_execution_failure_keeps_identity_and_error(self):
        self.dump(self.task/'result.json',dict(self.record,status='failed',error='interrupted'))
        v=evidence.read_record(self.root,self.record);self.assertEqual(v['status'],'failed');self.assertEqual(v['error'],'interrupted');self.assertNotIn('views',v)
    def test_wrong_window_rejected(self):
        self.dump(self.task/'result.json',dict(self.record,window='001',status='completed',passed=True))
        with self.assertRaisesRegex(ValueError,'identity'):evidence.read_record(self.root,self.record)

if __name__=='__main__':unittest.main()
