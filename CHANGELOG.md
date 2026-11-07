# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-08

First public release.

### Added
- Rust crates `erploc-linalg`, `erploc-core`, `erploc-est`, `erploc-place` (no external dependencies, no `unsafe`):
  single-frame resection (4-dof position and yaw, or 6-dof) with covariance, Levenberg-Marquardt with a Huber option,
  bundle adjustment with surveyed markers and its initialisation, camera Fisher information, per-corner validity and
  sigma scale, marker visibility with 2.5D obstacles, marker placement (greedy with 1-swap and pruning, and the IFT
  minimal-count search), a survey-marker triplet choice, and a site model (`ERPSITE1` tables) with zone statistics.
- Command line `erploc resect | ba | place | site | e2e` (JSON in, JSON out) with example inputs in `examples/cli/`.
- Python package `erploc` (abi3, Python 3.9+, numpy): `resection`, `bundle_adjust`, `init_bundle`, `fisher_camera`,
  `Scene`, `place_markers`, `evaluate`, `wall_candidates`, `observations_from_eqar` (duck typing, eqar is not a
  dependency), type stubs, GIL released during computation, bad input raises exceptions.
- Tests against outputs recorded from the Python reference implementation (`fixtures/`, 5 MB), self-checks, CLI
  sweeps, a random-input harness, Kani harnesses (`verify/`), Python tests.
- `examples/end_to_end.py` (synthetic room, eqar detection), `bench/bench.py` and `bench/RESULTS.md`,
  `docs/make_figures.py`.

### Changed before release
- `scene::cells` and `scene::candidates_from_walls` return `Result<_, PlaceError>`; `PlaceError::InvalidInput`, `ResectError::InvalidInput` and
  `LinalgError::Shape` are new. Bad arguments to `resect`, `optimize`, `optimize_ift`, `evaluate`, `local_search_1swap`, `prune`, `brute_force`,
  `ift_det`, `run_ift`, `brute_key` and `sigma_pos` are errors, not panics.
- A bundle adjustment whose dense Hessian would exceed 16 million entries is an input error (`io::MAX_BA_HESSIAN`).
- The occlusion dot-product and hit-test fixture families derived from a private site were removed; the hit paths are checked against a scalar
  enumeration on the synthetic site.
- Tests: exact selections and step counts only on macOS aarch64; elsewhere keys within a tolerance (`ERPLOC_TEST_LOOSE=1` selects those gates).
- `NOTICE` lists the licences of the crates linked into the binaries, with the full notices of rust-numpy, memoffset, zmij and memchr.

### Notes
- Field experiments have not been run; every accuracy figure in this release comes from synthetic data.
- Not included: the partial-occlusion credit model of the research code, which depends on measured data.
- Known limits (not met or not run) are listed in the README and in `verify/README.md`.
