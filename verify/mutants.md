# Mutation testing (`cargo-mutants`): NOT RUN

`cargo-mutants` has not been run on this code base, so there is no mutant count and no mutation score.

To close this gap: install `cargo-mutants`, run it on `erploc-linalg`, `erploc-core`, `erploc-est` and `erploc-place` with the test
command `cargo test --release`, and record the real totals here. The input-mutation sweeps of the CLI
(`cli_malformed::mutation_sweep_*`) and the random smoke harness (`erploc-cli/tests/fuzz_bodies.rs`) mutate inputs, not the source, and
are not a substitute.
