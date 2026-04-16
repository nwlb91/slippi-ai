# Changelog

This file tracks every modification to upstream files in this fork. Each entry should include the file path, approximate line range, and purpose of the change. This serves as the merge-conflict map when pulling from `upstream/main`.

New files under `slippi_ai/custom/` or `tests/custom/` do not need to be logged here — only edits to upstream files.

## Entries

- `slippi_db/parse_peppi.py` (`get_base_player_data`, ~line 75): Added per-frame `stocks` extraction to support tournament win rules and agent observations.
- `slippi_db/parse_libmelee.py` (`get_base_player`, ~line 45): Mirrored `stocks` extraction so the live-eval parser stays in sync with the peppi parser.
- `slippi_ai/types.py` (`Nana`, `Player` NamedTuples, ~line 45 / ~line 58): Added `stocks: np.uint8` field in both tuples.
- `slippi_db/preprocessing.py` (imports and `get_metadata` winner assignment, ~line 15 / ~line 214): Delegated winner determination to `slippi_ai.custom.tournament_rules.compute_correct_winner` so the 45-ledge-grab rule is applied; upstream `compute_winner` is preserved and passed as the fallback.
- `slippi_ai/embed.py` (new `embed_stocks`, `_base_player_embedding`, `make_player_embedding`, `PlayerConfig`, ~lines 426-510): Added an optional `stocks` one-hot embedding for the player observation.
