import numpy as np
from motiogen.native_analysis import reset_aware_arrival

def test_eventual_success_after_reset_does_not_count_before_reset():
    joints=np.ones((60,1,3));joints[40:,0,:2]=0
    result=reset_aware_arrival(joints,30,[0,0,0],10,[30])
    assert result['first_eventual_hold_frame']==40
    assert result['new_goal_success_before_next_reset']==0
    assert result['new_goal_success_reset_free_episode']==0

def test_reset_before_new_goal_does_not_invalidate_new_episode_arrival():
    joints=np.zeros((60,1,3))
    result=reset_aware_arrival(joints,30,[0,0,0],20,[10])
    assert result['new_goal_success_before_next_reset']==1
    assert result['new_goal_success_reset_free_episode']==0

def test_reset_during_hold_prevents_cross_episode_success():
    joints=np.ones((60,1,3));joints[20:40,0,:2]=0
    result=reset_aware_arrival(joints,30,[0,0,0],20,[30])
    assert result['first_eventual_hold_frame']==20
    assert result['first_hold_before_next_reset_frame'] is None
    assert result['new_goal_success_before_next_reset']==0
