import numpy as np
import pytest
from motiogen.native_analysis import geometric_contact_persistence

def test_no_contact_is_missing_retention_not_zero():
    r=geometric_contact_persistence(np.full((4,2),.2),30)
    assert r['retention_fraction'] is None
    assert r['eligible_hand_transitions']==0
    assert r['longest_observed_span_seconds']==0

def test_switching_hands_is_not_continuous_same_hand_contact():
    d=np.array([[.01,.2],[.2,.01],[.01,.2],[.2,.01]])
    r=geometric_contact_persistence(d,30)
    assert r['eligible_hand_transitions']==3 and r['retained_hand_transitions']==0
    assert r['retention_fraction']==0 and r['longest_observed_span_seconds']==0

def test_window_censoring_and_strict_threshold():
    d=np.array([[.01,.05],[.01,.05],[.01,.05],[.01,.05]])
    r=geometric_contact_persistence(d,30)
    assert r['retention_fraction']==1 and r['eligible_hand_transitions']==3
    assert r['longest_observed_span_seconds']==pytest.approx(.1)
    with pytest.raises(ValueError):geometric_contact_persistence(d,0)
