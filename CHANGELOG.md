# Changelog

This file tracks every modification to upstream files in this fork. Each entry should include the file path, approximate line range, and purpose of the change. This serves as the merge-conflict map when pulling from `upstream/main`.

New files under `slippi_ai/custom/` or `tests/custom/` do not need to be logged here — only edits to upstream files.

## Entries

- `slippi_db/parse_peppi.py` (`get_base_player_data`, ~line 75): Added per-frame `stocks` extraction to support tournament win rules and agent observations.
- `slippi_db/parse_libmelee.py` (`get_base_player`, ~line 45): Mirrored `stocks` extraction so the live-eval parser stays in sync with the peppi parser.
- `slippi_ai/types.py` (`Nana`, `Player` NamedTuples, ~line 45 / ~line 58): Added `stocks: np.uint8` field in both tuples.
- `slippi_db/preprocessing.py` (imports and `get_metadata` winner assignment, ~line 15 / ~line 214): Delegated winner determination to `slippi_ai.custom.tournament_rules.compute_correct_winner` so the 45-ledge-grab rule is applied; upstream `compute_winner` is preserved and passed as the fallback.
- `slippi_ai/embed.py` (new `embed_stocks`, `_base_player_embedding`, `make_player_embedding`, `PlayerConfig`, ~lines 426-510): Added an optional `stocks` one-hot embedding for the player observation.
- `slippi_ai/types.py` (`Nana`, `Player`, `Game` NamedTuples): Added `ledge_grabs: np.uint16` field on `Nana` and `Player` (after `stocks`), and `remaining_time: np.float32` field on `Game` (after `stage`), for agent observations.
- `slippi_db/parse_peppi.py` (`get_base_player_data`, `from_peppi`): Populate per-frame `ledge_grabs` via `cumulative_ledge_grabs` helper and compute the normalized `remaining_time` scalar from `peppi_game.start.timer`.
- `slippi_db/parse_libmelee.py` (`get_base_player`, `Parser`): Mirror `ledge_grabs` tracking per port across frames and compute `remaining_time` from `game.frame`; added a `timer_seconds` parameter to `Parser` (default 480s).
- `slippi_ai/embed.py` (new `embed_ledge_grabs`, `embed_remaining_time`, `_base_player_embedding`, `make_player_embedding`, `make_game_embedding`, `PlayerConfig`): Added optional `ledge_grabs` one-hot embedding for the player observation and `remaining_time` float embedding for the game observation.
- `slippi_ai/embed.py` (`embed_ledge_grabs`, ~line 435): Narrowed the one-hot embedding `dtype` from `np.uint16` to `np.uint8`. `tf.one_hot` rejects `uint16` indices, and 50 buckets fits trivially in `uint8`. The underlying `ledge_grabs` field in `types.py` remains `uint16`; `OneHotEmbedding.from_state` narrows it via CLAMP + astype before TF sees it.
- `slippi_ai/embed.py` (`embed_ledge_grabs`, ~line 435): Narrowed the one-hot embedding `dtype` from `np.uint16` to `np.uint8`. `tf.one_hot` rejects `uint16` indices, and 50 buckets fits trivially in `uint8`. The underlying `ledge_grabs` field in `types.py` remains `uint16`; `OneHotEmbedding.from_state` narrows it via CLAMP + astype before TF sees it.
