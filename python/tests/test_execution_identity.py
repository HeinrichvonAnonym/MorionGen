import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from motiogen.execution_identity import framework_sources,contract_values


class ExecutionIdentityTests(unittest.TestCase):
    def test_resume_binds_preprocessing_and_text_weights_beyond_plan_files(self):
        from motiogen.execution_identity import identity_payload
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            plan={'jobs':[{'method':'motionlcm','command':['python']}]}
            cfg={'baselines':{'motionlcm':{}}}
            with patch('motiogen.execution_identity.repo_root',return_value=root), \
                 patch('motiogen.execution_identity.subprocess.check_output',return_value=b''), \
                 patch('motiogen.weight_identity.snapshot',return_value={'identity':'weights1'}) as weights, \
                 patch('motiogen.preprocessing_identity.snapshot',return_value={'identity':'norm1'}) as preprocessing:
                first=identity_payload(plan,cfg,root)
                preprocessing.return_value={'identity':'norm2'}
                changed=identity_payload(plan,cfg,root)
                self.assertNotEqual(first,changed)
                preprocessing.return_value={'identity':'norm1'}
                weights.return_value={'identity':'weights2'}
                self.assertNotEqual(first,identity_payload(plan,cfg,root))

    def test_import_closure_excludes_report_and_follows_relative_imports(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'utils/motiogen';package.mkdir(parents=True)
            scripts=root/'python/scripts';scripts.mkdir(parents=True)
            (scripts/'worker.py').write_text('from motiogen.worker import run\n')
            (scripts/'report.py').write_text('print("unrelated")\n')
            (package/'worker.py').write_text('from . import decoder\n')
            (package/'decoder.py').write_text('import motiogen.weights\n')
            (package/'weights.py').write_text('value=1\n')
            plan={'jobs':[{'command':['python',str(scripts/'worker.py')]}]}
            files={p.name for p in framework_sources(plan,root)}
            self.assertEqual(files,{'worker.py','decoder.py','weights.py'})

    def test_report_contract_does_not_change_execution_identity(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config').mkdir()
            (root/'config/metrics.json').write_text('{"threshold":0.05}')
            report=root/'config/survey_migration.json';report.write_text('{"title":"one"}')
            registry=root/'config/native_component_experiments.json'
            registry.write_text(json.dumps({'artifact_policy':{'eager':False},'methods':{'description':'one'}}))
            first=contract_values(root)
            report.write_text('{"title":"two"}')
            registry.write_text(json.dumps({'artifact_policy':{'eager':False},'methods':{'description':'two'}}))
            self.assertEqual(first,contract_values(root))
            (root/'config/metrics.json').write_text('{"threshold":0.10}')
            self.assertNotEqual(first,contract_values(root))

if __name__=='__main__':unittest.main()
