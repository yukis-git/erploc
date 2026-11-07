//! End-to-end checks E2E-1..4 (release only; the debug-mode formula tests live in `src/e2e.rs`).
use erploc_cli::e2e::{median, run_e2e_info, CheckLine, E2eConsts, E2eInfo, PY_IFT_BUDGET_S};
use erploc_place::search::Budget;
use std::process::Command;

/// Python values cited from erploc/out/e2e_huber/e2e_selfcheck.txt (seed 20260912).
const PY_CITED: [&str; 6] = [
    "mean(d^2)/3=1.0218, ratio=1.0253, 3sigma=0.9986, max|e|=0.0103 m",
    "max rel=3.98e-14, mismatch=0",
    "mean(d^2_full)/3=0.8810, ratio=0.9350, inflation=5.7624",
    "frac=0.9986, ratio=1.2304, max|e|=0.7875 m, single=94",
    "ratio=3.2372",
    "39.9 s",
];

fn run(seed: u64) -> (Vec<CheckLine>, E2eInfo) {
    let c = E2eConsts { seed, ..E2eConsts::default() };
    run_e2e_info(&c, &Budget::Wall(PY_IFT_BUDGET_S)).expect("e2e run")
}

fn print_table(tag: &str, lines: &[CheckLine], info: &E2eInfo) {
    println!("=== {tag}: sel={:?} survey={:?} key=({}, {:.6}, {:.6})", info.sel, info.survey_ids, info.key.0, info.key.1, info.key.2);
    println!(
        "timings [s]: ift={:.2} e1={:.2} e2={:.2} e3={:.2} e4={:.2} total={:.2}",
        info.t_ift, info.t_e1, info.t_e2, info.t_e3, info.t_e4, info.t_total
    );
    for (l, p) in lines.iter().zip(PY_CITED) {
        println!(
            "[{}] {}\n    Rust:     {}\n    Python:   {}\n    criterion: {}",
            if l.passed { "pass" } else { "FAIL" },
            l.name,
            l.measured,
            p,
            l.criterion
        );
    }
}

/// Asserts E2E-1, 1b, 2, 4, the run-time line and E2E-3's fraction criterion (frac >= 0.95). The E2E-3 RATIO is printed and
/// labelled KNOWN-FRAGILE (Python criterion unstable across seeds, see the e2e module docs); it is asserted as a distribution in
/// `e2e3_ratio_distribution`, not at a fixed seed.
fn assert_known_fragile(lines: &[CheckLine], info: &E2eInfo) {
    println!(
        "E2E-3 ratio={:.4} (window [0.8, 1.25]: {}) KNOWN-FRAGILE (Python criterion; see the e2e module docs); frac={:.4}",
        info.e3_ratio,
        if (0.8..=1.25).contains(&info.e3_ratio) { "inside" } else { "OUTSIDE" },
        info.e3_frac
    );
    for (i, l) in lines.iter().enumerate() {
        if i == 3 {
            continue;
        }
        // the 90 s limit is a wall-clock criterion: off the reference platform (or under the libm shim of the loose-gate check) it
        // measures the machine, not the code
        if l.name == "e2e run time" && !erploc_testkit::strict() {
            continue;
        }
        assert!(l.passed, "{}: {}", l.name, l.measured);
    }
    assert!(info.e3_frac >= 0.95, "E2E-3 frac {}", info.e3_frac);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only")]
fn e2e_all_pass() {
    let (lines, info) = run(E2eConsts::default().seed);
    print_table("seed 20260912", &lines, &info);
    assert_eq!(lines.len(), 6);
    assert_known_fragile(&lines, &info);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only")]
fn e2e_seed_robustness() {
    let base = E2eConsts::default().seed;
    for d in [7u64, 13] {
        let (lines, info) = run(base + d);
        print_table(&format!("seed SEED+{d}"), &lines, &info);
        assert_known_fragile(&lines, &info);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only")]
fn e2e_1b_exact_model() {
    let (lines, _) = run(E2eConsts::default().seed);
    let l = &lines[1];
    println!("{}\nPython: max rel 3.98e-14, mismatch 0", l.measured);
    // "max relative difference=3.98e-14, visible-set mismatches=0, ..."
    let rel: f64 = l.measured.split("difference=").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    let mm: usize = l.measured.split("mismatches=").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    assert!(rel < 1e-6, "{}", l.measured);
    assert_eq!(mm, 0, "{}", l.measured);
}

/// `erploc e2e` prints the lines and exits 0 (all pass) or 1 (a check failed); never 3/4/70. No budget flag is passed
/// (the default 120 s IFT budget applies); the table is the full pipeline, so release only.
#[test]
#[cfg_attr(debug_assertions, ignore = "release only")]
fn e2e_cli_smoke() {
    let o = Command::new(env!("CARGO_BIN_EXE_erploc")).args(["e2e", "--seed-base", "20260912"]).output().expect("spawn");
    let code = o.status.code().expect("exit code");
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(code == 0 || code == 1, "exit {code}: {}", String::from_utf8_lossy(&o.stderr));
    assert!(out.contains("E2E-1 ") && out.contains("E2E-4") && out.contains("pass"), "{out}");
    assert_eq!(out.lines().filter(|l| l.contains(" | ")).count(), 6, "{out}");
}

#[test]
fn e2e_cli_usage_errors() {
    let bin = env!("CARGO_BIN_EXE_erploc");
    for args in [
        &["e2e", "--seed-base", "x"][..],
        &["e2e", "--nope", "1"],
        &["e2e", "--budget-s", "0"],
        &["e2e", "--seed-base", "18446744073709551615"],
    ] {
        let o = Command::new(bin).args(args).output().expect("spawn");
        assert_eq!(o.status.code(), Some(2), "{args:?}");
    }
}

/// Reference record: the Rust IFT selection against the Python-class result `[6, 7, 1, 2]` (and its x -> 8-x mirror `[6, 5, 1, 0]`).
/// Tie-equivalent only if the keys are equal.
#[test]
fn e2e_selection_key_vs_python_class() {
    use erploc_place::placement::{demo_spec, evaluate, key3, precompute};
    let spec = demo_spec(10);
    let tb = precompute(&spec).unwrap();
    let key_of = |sel: &[usize]| {
        let k = key3(&evaluate(&tb, sel, &spec).unwrap(), &spec);
        (k.n_infeasible, k.penalty_sum, k.neg_margin)
    };
    let rust = erploc_place::search::optimize_ift(&spec, 0.005, &Budget::Wall(PY_IFT_BUDGET_S)).unwrap();
    let kr = key_of(&rust.sel);
    println!("Rust sel={:?} key={kr:?}", rust.sel);
    println!("[6,7,1,2] key={:?}", key_of(&[6, 7, 1, 2]));
    println!("[6,5,1,0] key={:?}", key_of(&[6, 5, 1, 0]));
    println!("survey={:?}", rust.survey_ids);
    if erploc_testkit::strict() {
        assert_eq!(rust.sel, vec![1, 4, 6, 9]);
        assert_eq!(rust.survey_ids, vec![4, 6, 9]);
    }
    assert!(kr.0 == 0 && kr.2 < key_of(&[6, 7, 1, 2]).2, "{kr:?}");
}

/// E2E-3 ratio over SEED..SEED+30. Python sweep of the unmodified check: 9 of 31 seeds miss
/// [0.8, 1.25] (22/31 pass), median 0.992, range 0.775-1.931. E2E-3 does not depend on the E2E-2 sizes, so a cut-down run
/// (1 map, 4 test frames) is used.
#[test]
#[cfg_attr(debug_assertions, ignore = "release only")]
fn e2e3_ratio_distribution() {
    let base = E2eConsts::default().seed;
    let mut ratios = Vec::new();
    for d in 0..31u64 {
        let c = E2eConsts { seed: base + d, n_map_realizations: 1, n_test_ba: 4, ..E2eConsts::default() };
        let (_, info) = run_e2e_info(&c, &Budget::Wall(PY_IFT_BUDGET_S)).expect("e2e run");
        assert!(info.e3_frac >= 0.95, "seed +{d}: frac {}", info.e3_frac);
        ratios.push(info.e3_ratio);
    }
    let pass = ratios.iter().filter(|r| (0.8..=1.25).contains(*r)).count();
    let med = median(&ratios);
    let (mn, mx) = ratios.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &r| (a.min(r), b.max(r)));
    println!(
        "E2E-3 ratio over 31 seeds: pass {pass}/31, median {med:.4}, min {mn:.4}, max {mx:.4} (Python: 22/31, median 0.992, 0.775-1.931)"
    );
    assert!(pass >= 15, "pass count {pass}/31");
    assert!((0.85..=1.20).contains(&med), "median {med}");
}
