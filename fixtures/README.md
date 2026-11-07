# Test fixtures

Plain-text records `<tag> <name> <rows> <cols> <values...>` (Python `repr` floats, round-trip exact; booleans 0/1). Each file is
read by the differential tests of one crate; every test ends by checking that all records of its group were read.

| File | Written by | Read by |
|---|---|---|
| `linalg.txt` | `tools/export_linalg_fixtures.py` (numpy only) | `erploc-linalg` |
| `core.txt` | `tools/export_core_fixtures.py` | `erploc-core` |
| `est.txt` | `tools/export_est_fixtures.py` | `erploc-est`, and `cli/` below |
| `place.txt` | `tools/export_place_fixtures.py` | `erploc-place` |
| `typed.txt` | `tools/export_typed_fixtures.py` | `erploc-place` (`oc_typed`) |
| `ift.txt` | recorded from the Python reference with private tooling (thinned; not regenerable from this repository) | `erploc-place` (`diff_site`) |
| `cli/*.json` | `tools/export_cli_cases.py` (from `est.txt` and `place.txt`) | `erploc-cli` |

The exporters need the Python reference implementation (the `erploc` package) and numpy 2.0.2; set `ERPLOC_PY_ROOT` to the directory that
contains the package. Output is deterministic. The records were thinned to keep the total under 5 MB: fewer frames per
bundle-adjustment problem, a coarser grid in the second placement room, fewer random cases in the geometry, sigma and zone families.
All case kinds and tolerances are unchanged. The occlusion dot-product and hit families (`oc_dot3_*`, `oc_hit_random`) were removed
because they were derived from a private site; the occlusion dot and hit paths therefore have no recorded Python reference in public (the
tests compare the fast and scalar implementations on the synthetic site instead).
