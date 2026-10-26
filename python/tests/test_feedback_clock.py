import time
import numpy as np
from motiogen.feedback import run_episode


def test_goal_ablation_preserves_history_and_changes_only_goal_input():
    goals = [{'frame': 0, 'position': [1]}, {'frame': 3, 'position': [2]}]
    observations = []
    def generate(obs):
        observations.append(obs)
        return np.repeat(obs.goal[None], 2, axis=0)
    result, trace = run_episode(generate, [[0], [0]], goals, fps=10, seconds=.6,
                                lookahead_seconds=0)
    assert result[-1, 0] == 2
    assert observations[-1].human_history.shape == (2, 1)
    frozen, _ = run_episode(generate, [[0], [0]], goals, fps=10, seconds=.6,
                            feedback=False, lookahead_seconds=0)
    assert np.all(frozen == 1)
    assert trace['hold_frames'] == 0


def test_realtime_counts_compute_delay_and_expired_predictions():
    def slow(obs):
        time.sleep(.06)
        return np.arange(20, dtype=float)[:, None]
    _, trace = run_episode(slow, [[0], [0]], [{'frame': 0, 'position': [1]}],
                           fps=100, seconds=.2, mode='realtime', lookahead_seconds=0, prefill=False)
    assert trace['wall_seconds'] >= .19
    assert trace['hold_frames'] >= 5
    assert trace['requests'][0]['expired_frames'] >= 5


def test_committed_handoff_continues_forecast_without_overwriting_queue():
    observed = []
    def generate(obs):
        observed.append(obs)
        return obs.human_history[-1] + np.arange(1, 7)[:, None]
    result, trace = run_episode(generate, [[-1], [0]], [{'frame': 0, 'position': [1]}],
        fps=10, seconds=2, lookahead_seconds=.3, handoff='committed')
    np.testing.assert_array_equal(result[:, 0], np.arange(1, 21))
    assert observed[1].history_kind == 'committed_forecast'
    assert observed[1].frame > observed[1].observed_frame
    assert observed[1].human_history[-1, 0] > observed[1].observed_history[-1, 0]
    assert trace['protocol_version'] == 3


def test_committed_realtime_handoff_avoids_discarding_valid_prediction():
    def generate(obs):
        time.sleep(.02)
        return obs.human_history[-1] + np.arange(1, 11)[:, None]
    result, trace = run_episode(generate, [[-1], [0]], [{'frame': 0, 'position': [1]}],
        fps=50, seconds=.6, lookahead_seconds=.08, mode='realtime', handoff='committed')
    np.testing.assert_array_equal(result[:, 0], np.arange(1, 31))
    assert trace['hold_frames'] == 0
    assert all(x['expired_frames'] == 0 for x in trace['requests'])
