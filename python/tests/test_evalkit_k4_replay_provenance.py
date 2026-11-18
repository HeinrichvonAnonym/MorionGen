"""Original-solver replay must pin implementation and exact control settings."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/evalkit_k4_trajectory_replay.py'
spec=importlib.util.spec_from_file_location('replay',SCRIPT)
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Provenance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'solver.py';self.path.write_text('def solve(*args): return 42\n')
        self.settings={'max_nfev':20}
        self.source={'key':{'implementation':{'trajectory_solver.py':digest(self.path)},'config':{'trajectory_refinement':self.settings}}}
    def test_changed_solver_rejected_before_import(self):
        self.path.write_text('raise RuntimeError("must not execute")\n')
        with self.assertRaisesRegex(ValueError,'differs'):replay.install_original_solver(self.path,self.source,self.settings,digest)
    def test_candidate_settings_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'exact source'):replay.install_original_solver(self.path,self.source,dict(self.settings,linear_damping=.01),digest)
    def test_exact_solver_installed_and_receipted(self):
        solver=types.SimpleNamespace(solve=None)
        package=types.ModuleType('motiogen.evalkit.k');package.trajectory_solver=solver
        with patch.dict(sys.modules,{'motiogen.evalkit.k':package}):
            receipt=replay.install_original_solver(self.path,self.source,self.settings,digest)
        self.assertEqual(solver.solve(),42);self.assertEqual(receipt['sha256'],digest(self.path))

if __name__=='__main__':unittest.main()
