"""Tournament-accurate win labeling for Melee replays.

Implements the 45-ledge-grab rule: if a game ends by timeout and exactly one
player has 45+ ledge grabs, that player loses regardless of stocks/percent.
"""
from typing import Optional

import numpy as np

# Raw Slippi action state ID. Corresponds to melee.enums.Action.EDGE_CATCHING.
# Hardcoded here to avoid a runtime dependency on libmelee for post-processing.
EDGE_CATCHING_ACTION_ID = 252

LEDGE_GRAB_LIMIT = 45

# Melee runs at 60 FPS. The timer in metadata is in seconds.
FRAMES_PER_SECOND = 60

# Melee's frame index starts at -123. Frame 0 is when the timer begins counting.
GAME_START_FRAME = 0


def _entry_mask(action_array: np.ndarray) -> np.ndarray:
  """Boolean mask of frames that are a transition into EDGE_CATCHING.

  Frame 0 is also flagged if the action array starts in EDGE_CATCHING.
  """
  is_edge = action_array == EDGE_CATCHING_ACTION_ID
  entered = np.zeros_like(is_edge, dtype=np.bool_)
  if len(action_array) == 0:
    return entered
  entered[0] = is_edge[0]
  if len(action_array) > 1:
    entered[1:] = is_edge[1:] & ~is_edge[:-1]
  return entered


def count_ledge_grabs(action_array: np.ndarray) -> int:
  """Count transitions into EDGE_CATCHING across the frame array.

  A ledge grab is counted once per entry into EDGE_CATCHING, not per frame
  spent in it (EDGE_CATCHING lasts 7 frames for most characters).
  """
  return int(np.sum(_entry_mask(action_array)))


def cumulative_ledge_grabs(action_array: np.ndarray) -> np.ndarray:
  """Per-frame cumulative count of ledge grabs.

  Output has the same length as the input and is a running count of
  transitions into EDGE_CATCHING up to and including each frame.
  """
  return np.cumsum(_entry_mask(action_array)).astype(np.uint16)


def game_ended_by_timeout(last_frame: int, timer_seconds: int) -> bool:
  """Return True if the game reached its time limit.

  Use a small tolerance: a game that ends within the final second is a timeout.
  """
  total_game_frames = timer_seconds * FRAMES_PER_SECOND
  return last_frame >= total_game_frames - FRAMES_PER_SECOND


def compute_correct_winner(
    last_frame: int,
    timer_seconds: int,
    action_arrays: list,   # list of per-player action np.ndarrays
    final_stocks: list,    # list of per-player int stock counts (last frame)
    final_percents: list,  # list of per-player int percent values (last frame)
    upstream_winner: Optional[int],
) -> Optional[int]:
  """Apply tournament rules to determine the winner.

  Args:
      last_frame: index of the last frame of the game (0-based from timer start).
      timer_seconds: total game timer in seconds (typically 480 for tournament).
      action_arrays: per-player numpy arrays of action states.
      final_stocks: per-player stock count on the last frame.
      final_percents: per-player percent on the last frame.
      upstream_winner: the winner index computed by upstream's logic.

  Returns:
      Player index of the winner, or None if the game cannot be resolved
      (e.g., truly tied under all rules).
  """
  if not game_ended_by_timeout(last_frame, timer_seconds):
    return upstream_winner

  ledge_grabs = [count_ledge_grabs(a) for a in action_arrays]
  over_limit = [g >= LEDGE_GRAB_LIMIT for g in ledge_grabs]

  # Rule: if exactly one player is over the limit, that player loses.
  if sum(over_limit) == 1:
    loser_idx = over_limit.index(True)
    # Two-player case. Generalize if teams support is added later.
    if len(final_stocks) == 2:
      return 1 - loser_idx
    return upstream_winner  # Fall back for non-2-player edge cases.

  # Otherwise, normal timeout rules: higher stocks wins; if tied, lower percent.
  if len(final_stocks) != 2:
    return upstream_winner

  if final_stocks[0] != final_stocks[1]:
    return 0 if final_stocks[0] > final_stocks[1] else 1
  if final_percents[0] != final_percents[1]:
    return 0 if final_percents[0] < final_percents[1] else 1
  return None  # True tie.
