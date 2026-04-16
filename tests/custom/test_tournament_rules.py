"""Tests for slippi_ai.custom.tournament_rules."""
import unittest

import numpy as np

from slippi_ai.custom import tournament_rules as tr


EDGE = tr.EDGE_CATCHING_ACTION_ID
STAND = 14  # An arbitrary non-edge-catching action state.


class CountLedgeGrabsTest(unittest.TestCase):

  def test_zero_grabs(self):
    actions = np.array([STAND] * 100, dtype=np.uint16)
    self.assertEqual(tr.count_ledge_grabs(actions), 0)

  def test_single_grab(self):
    # One transition STAND -> EDGE, held for several frames.
    actions = np.array([STAND] * 10 + [EDGE] * 7 + [STAND] * 10, dtype=np.uint16)
    self.assertEqual(tr.count_ledge_grabs(actions), 1)

  def test_multiple_grabs(self):
    # Two separate ledge grabs.
    actions = np.array(
        [STAND] * 5 + [EDGE] * 7 + [STAND] * 5 + [EDGE] * 7 + [STAND] * 5,
        dtype=np.uint16,
    )
    self.assertEqual(tr.count_ledge_grabs(actions), 2)

  def test_starts_in_edge_catching(self):
    # First frame is already EDGE_CATCHING; counts as one grab.
    actions = np.array([EDGE] * 7 + [STAND] * 10, dtype=np.uint16)
    self.assertEqual(tr.count_ledge_grabs(actions), 1)

  def test_empty_array(self):
    actions = np.array([], dtype=np.uint16)
    self.assertEqual(tr.count_ledge_grabs(actions), 0)

  def test_single_frame_edge(self):
    actions = np.array([EDGE], dtype=np.uint16)
    self.assertEqual(tr.count_ledge_grabs(actions), 1)

  def test_cumulative(self):
    actions = np.array(
        [STAND] * 3 + [EDGE] * 4 + [STAND] * 2 + [EDGE] * 3,
        dtype=np.uint16,
    )
    cum = tr.cumulative_ledge_grabs(actions)
    # Cum counts increment at each entry: 0*3, 1*4, 1*2, 2*3
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint8)
    np.testing.assert_array_equal(cum, expected)


class ComputeWinnerTest(unittest.TestCase):

  def _timeout_last_frame(self, timer_seconds: int = 480) -> int:
    # Any frame at or above total_frames - FPS counts as timeout.
    return timer_seconds * tr.FRAMES_PER_SECOND - 1

  def test_non_timeout_returns_upstream(self):
    # Early KO: last_frame well before timeout.
    last_frame = 60 * 60  # one minute in
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[np.array([STAND], dtype=np.uint16)] * 2,
        final_stocks=[0, 3],
        final_percents=[0, 50],
        upstream_winner=1,
    )
    self.assertEqual(winner, 1)

  def test_timeout_neither_over_limit_stocks_decide(self):
    last_frame = self._timeout_last_frame()
    p0_actions = np.array([STAND] * 1000, dtype=np.uint16)
    p1_actions = np.array([STAND] * 1000, dtype=np.uint16)
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[p0_actions, p1_actions],
        final_stocks=[2, 1],
        final_percents=[50, 30],
        upstream_winner=0,
    )
    self.assertEqual(winner, 0)

  def test_timeout_neither_over_limit_percent_tiebreak(self):
    last_frame = self._timeout_last_frame()
    actions = np.array([STAND] * 1000, dtype=np.uint16)
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[actions, actions],
        final_stocks=[2, 2],
        final_percents=[80, 40],
        upstream_winner=None,
    )
    # Lower percent wins the tiebreak.
    self.assertEqual(winner, 1)

  def test_timeout_exactly_one_over_limit_that_player_loses(self):
    last_frame = self._timeout_last_frame()
    # p0 grabs the ledge 50 times; p1 never grabs.
    p0_actions = np.concatenate([
        np.tile([STAND, STAND, EDGE], 50).astype(np.uint16),
        np.full(100, STAND, dtype=np.uint16),
    ])
    p1_actions = np.full(1000, STAND, dtype=np.uint16)
    self.assertGreaterEqual(tr.count_ledge_grabs(p0_actions), 45)
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[p0_actions, p1_actions],
        final_stocks=[4, 1],  # p0 has more stocks but should still lose
        final_percents=[0, 120],
        upstream_winner=0,
    )
    self.assertEqual(winner, 1)

  def test_timeout_both_over_limit_normal_rules(self):
    last_frame = self._timeout_last_frame()
    many_grabs = np.tile([STAND, EDGE], 50).astype(np.uint16)
    self.assertGreaterEqual(tr.count_ledge_grabs(many_grabs), 45)
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[many_grabs, many_grabs.copy()],
        final_stocks=[3, 1],
        final_percents=[50, 50],
        upstream_winner=0,
    )
    # Both over limit => normal rules => higher stocks wins.
    self.assertEqual(winner, 0)

  def test_timeout_true_tie_returns_none(self):
    last_frame = self._timeout_last_frame()
    actions = np.array([STAND] * 1000, dtype=np.uint16)
    winner = tr.compute_correct_winner(
        last_frame=last_frame,
        timer_seconds=480,
        action_arrays=[actions, actions],
        final_stocks=[2, 2],
        final_percents=[55, 55],
        upstream_winner=None,
    )
    self.assertIsNone(winner)


if __name__ == '__main__':
  unittest.main()
