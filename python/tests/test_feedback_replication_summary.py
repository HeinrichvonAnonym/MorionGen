import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('replication_summary',Path(__file__).parents[1]/'scripts/summarize_feedback_replication.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_single_hit_is_not_sustained():
    errors=np.ones(60)
    errors[10]=.1
    assert module.sustained(errors,.3,0)==(True,10/30)
    assert module.sustained(errors,.3,.5)==(False,None)


def test_onset_is_start_not_end_of_hold():
    errors=np.ones(60)
    errors[12:27]=.2
    assert module.sustained(errors,.3,.5)==(True,.4)
    assert module.sustained(errors,.3,1)==(False,None)


def test_boundary_and_censoring():
    assert module.sustained(np.full(15,.3),.3,.5)==(True,0.)
    assert module.sustained(np.full(14,.3),.3,.5)==(False,None)
