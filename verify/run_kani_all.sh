#!/bin/bash
# Run the Kani harnesses one at a time. Set WRAPPER to a command that prefixes each run (for example a memory gate); empty by default.
#   verify/run_kani_all.sh                       all harnesses
#   verify/run_kani_all.sh --only a,b,c          only the named harnesses
#   verify/run_kani_all.sh --list                print "crate harness" pairs
# Each run is a child of this script; its PID is written to $LOGDIR/current.pid while it runs and the script
# only ever signals that one PID (and its own children on INT/TERM). It never pattern-kills (no pkill/killall).
# Logs: $LOGDIR/<harness>.log (default: target/kani-logs (git-ignored), override with KANI_LOGDIR). Summary line per harness on stdout.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPPER="${WRAPPER:-}"
LOGDIR="${KANI_LOGDIR:-$ROOT/target/kani-logs}"
mkdir -p "$LOGDIR"

# "crate harness" pairs
HARNESSES="
erploc-linalg matmul_shapes
erploc-linalg lu_2x2_total
erploc-linalg lu_2x2_solve_len
erploc-linalg cholesky_2x2_total
erploc-linalg eigh_2x2_sorted
erploc-linalg unit_interval
erploc-core wrap_pi_range
erploc-core erp_camera_new_valid
erploc-core erp_camera_new_invalid
erploc-core residual_wrap_bound
erploc-est huber_weight
erploc-est lm_advance_inductive
erploc-est lm_drive_any_sequence
erploc-est lm_float_narrow
erploc-est ba_layout_assembly_indices
erploc-est scatter_block_indices
erploc-est schur_slicing_indices
erploc-est prop_key_total_order
erploc-est masked_row_map
erploc-place np_mod_range
erploc-place lam_in_core_total
erploc-place arange_len_bounds
erploc-place ravel_xy_in_bounds
erploc-place round_half_even_props
erploc-place table_indices_in_bounds
erploc-place sigma10_branches
erploc-place sigma10_finish_range
erploc-place sigma10_finish_nan
erploc-place cap_mapping
erploc-place ift_pair_key_large
erploc-place ift_pair_key_small
erploc-place ift_csr_slicing
erploc-place ift_vis_csr
erploc-place ift_pick_min
erploc-place better_margin_tol_asymmetric
erploc-place better_pen_eq_symmetric
erploc-place better_asymmetric
erploc-place better_ordinal_end_to_end
"

ONLY=""
case "${1:-}" in
  --list) echo "$HARNESSES" | sed '/^$/d'; exit 0 ;;
  --only) ONLY=",${2:?--only needs a comma-separated list},"; ;;
  "") ;;
  *) echo "usage: $0 [--list | --only h1,h2,...]" >&2; exit 2 ;;
esac

CHILD=""
on_signal() {
  # only our own child
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  exit 143
}
trap on_signal INT TERM

cd "$ROOT" || exit 1
fail=0
while read -r crate h; do
  if [ -n "$ONLY" ]; then case "$ONLY" in *",$h,"*) ;; *) continue ;; esac; fi
  log="$LOGDIR/$h.log"
  start=$(date +%s)
  $WRAPPER cargo kani -p "$crate" --harness "$h" >"$log" 2>&1 &
  CHILD=$!
  echo "$CHILD" >"$LOGDIR/current.pid"
  wait "$CHILD"; rc=$?
  CHILD=""; rm -f "$LOGDIR/current.pid"
  end=$(date +%s)
  res=$(grep -E '^VERIFICATION:-' "$log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
  [ "$rc" -eq 0 ] || fail=1
  echo "$crate $h rc=$rc $((end-start)) s ${res:-no-verdict}"
done < <(echo "$HARNESSES" | sed '/^$/d')
exit $fail
