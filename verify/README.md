# Verification record

What backs the claims about correctness, and what does not.

## Tests

`pytest tests/py` runs the Python tests (numpy and pytest only): resection against the truth on synthetic frames, covariance and a Monte-Carlo
check of `sigma_pos`, 6-dof, per-corner masks, bundle adjustment with surveyed markers, placement, `evaluate` against `place_markers`, the eqar bridge,
bad-input and random-input sweeps (the outcome is always an ordinary exception), and a check that the GIL is released.

`cargo test --release` runs the unit tests of every crate and the integration tests:

- **Differential tests** compare against outputs recorded from the Python reference implementation (`fixtures/*.txt`, see
  `fixtures/README.md`): counts, masks and selections exactly; floats within the tolerance named in each test (rotation, Lie group and
  marker algebra 1e-12, least squares 1e-9, bundle adjustment states 1e-8..1e-9, placement keys 1e-9 relative).
- **Self-checks** ported from the Python modules: Fisher information against the Monte-Carlo covariance (ratio within 5 %),
  resection and bundle-adjustment statistics, gauge null space, additivity of the placement information table.
- **Site model** tests use the synthetic table of `erploc_place::site_synth`: loader round trip and corruption handling (every prefix and
  a byte-flip sweep), the fast evaluator against the from-scratch evaluator (bit-level on counts, 1e-9 on penalties), searches, zone statistics.
- **CLI** tests: exit codes, round trips against the recorded Python outputs, malformed-input sweeps (never exit 70), a random-smoke fuzz
  harness (`crates/erploc-cli/tests/fuzz_bodies.rs`, 5000 inputs per target by default, `ERPLOC_FUZZ_N` to raise it).

The fixtures were recorded with numpy on Apple aarch64, and the Rust results were compared there first. Comparisons against the fixtures
are exact only for counts, masks, selections (ties within a stated band) and for values that are pure `+ - * /` arithmetic, which Rust does
not fuse (areas, grids, dot products); everything that passes through `sin`, `cos`, `atan2`, `acos` is compared with the tolerances above, so a
different libm or a build without fused multiply-add does not change a verdict unless a threshold decision (a visibility mask, a feasibility
flag) falls within an ulp of its threshold. The comparisons in the tests that run the library against itself are bit-level in one process.
**CI runs on Linux x86_64 and Windows x64 as well; those runs were not available while this was prepared (the author's machine is
aarch64), so nothing here claims they pass.** The tests therefore have two sets of gates (`erploc_testkit::strict()`): on the
reference platform (macOS, aarch64) the selections, step counts and tight float bounds are exact as recorded; everywhere else the runs of
a search are compared by their objective key within a tolerance (ties may resolve differently), and the estimator and covariance
comparisons get 100 times the tolerance (`bi_noisy` marker R about 1e-7; `pinv_hybrid_accuracy` 0.5 instead of 0.1). `ERPLOC_TEST_LOOSE=1`
selects the off-reference gates on any platform. They were exercised on the reference machine with a throwaway libm shim (a C file loaded with `DYLD_INSERT_LIBRARIES`, not part of the repository) that moves
the result of `sin`, `cos`, `atan2`, `hypot`, `acos`, ... by one ulp in a pseudo-random 1 % and 5 % of the calls: with `ERPLOC_TEST_LOOSE=1`, the whole `cargo test --release` (465 tests) passed at 1 % and at 5 %, each with four shim seeds (1 to 4).
Without `ERPLOC_TEST_LOOSE=1` the same shim fails the exact gates (selection and step-count comparisons), which is what the loose gates are for.
The shim also moves `hypot(x, 0)`, which real libms return exactly, and so it is harsher than a real libm on the edge-length rounding of the wall
candidates; one comparison (`sc_cands`, an edge whose length over the spacing is exactly a half integer) is skipped off the reference platform for that reason.
The run-time line of `erploc e2e` (< 90 s) is not asserted off the reference platform.

`cargo +nightly miri test -p <crate> --lib` with `MIRIFLAGS=-Zmiri-no-extra-rounding-error` was clean for `erploc-place` and `erploc-est` and
the CLI unit tests at the time of the last full run on the earlier code base. Tests that read fixture files or run a pipeline are
`#[cfg_attr(miri, ignore)]`. Not re-run after the extraction.

## Known limits (NOT MET / NOT RUN)

- **`erploc e2e` exits 0: NOT MET.** The E2E-3 ratio `RMS(|e|)/RMS(sigma_pos)` in [0.8, 1.25] is unstable across seeds in the Python
  reference itself (it misses on 9 of 31 seeds, median 0.992, range 0.775..1.931). The criterion is unchanged in Python and Rust; the Rust
  command can exit 1 at the default seed. `e2e3_ratio_distribution` asserts the distribution over 31 seeds instead (19 of 31 passed in the
  recorded run, median 1.14). The other checks pass.
- **IFT neighbourhood evaluation time** is measured on the synthetic site only (92 cells, 54 candidates): median 0.89 ms for a nine-marker
  selection, below the 5 ms design target (`bench/RESULTS.md` part C). No measurement exists for scenes with hundreds of candidates, so the
  target is not claimed for them.
- **Coverage-guided fuzzing: NOT RUN.** The random smoke harness above replaces it; it cannot find inputs a coverage-guided fuzzer would reach.
- **Mutation testing (`cargo-mutants`): NOT RUN.** No mutation score exists.
- **Python parity of the occlusion hit paths** (`hit_board`, `hit_base`, the dot products inside them, blocked triples on a stand-bearing site)
  has no recorded Python reference in public. The public tests keep the primitive-level face-sample and `precompute_typed` comparisons against
  recorded Python output, and `oc_hit_self_check` compares the fast blocked-triple construction with an unfiltered scalar enumeration on the
  synthetic site. That is a consistency check between two Rust paths, not a comparison with Python.
- **Partial-occlusion credit** (`classify`, capability tables, `rescore`) is not part of this release: it depends on a static raster of
  a measured scene and on detector measurements that cannot be published.
- **MSRV**: Rust 1.80.1 builds and passes `cargo test --release --locked` for the four computation crates and the CLI (all targets, including
  the tests). The Python extension crate is built with the current stable toolchain only.
