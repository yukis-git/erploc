//! Exit codes of the `erploc` binary, observed from outside the process.
use std::process::{Command, Output};

fn erploc(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_erploc")).args(args).output().expect("spawn erploc")
}

fn fixture(name: &str) -> String {
    format!("{}/../../fixtures/cli/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exit code (not killed by a signal)")
}

#[test]
fn success_is_zero_and_prints_json() {
    let o = erploc(&["resect", "--in", &fixture("resect_yaw_0.json")]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let out = String::from_utf8(o.stdout).unwrap();
    assert!(out.starts_with('{') && out[..400.min(out.len())].contains("\"converged\""), "{}", &out[..200.min(out.len())]);
}

#[test]
fn out_and_csv_files() {
    let dir = std::env::temp_dir().join(format!("erploc_cli_exit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (j, c) = (dir.join("r.json"), dir.join("r.csv"));
    let o = erploc(&["place", "--in", &fixture("place_demo_0.json"), "--out", j.to_str().unwrap(), "--csv", c.to_str().unwrap()]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert!(std::fs::read_to_string(&j).unwrap().contains("\"sel\""));
    assert!(std::fs::read_to_string(&c).unwrap().starts_with("id,x,y,z,azimuth_deg,surveyed\n"));
    // unwritable destinations are I/O errors
    let bad = dir.join("no_such_dir").join("x.json");
    let o = erploc(&["resect", "--in", &fixture("resect_yaw_0.json"), "--out", bad.to_str().unwrap()]);
    assert_eq!(code(&o), 5);
    let o = erploc(&["place", "--in", &fixture("place_demo_0.json"), "--csv", bad.to_str().unwrap(), "--out", j.to_str().unwrap()]);
    assert_eq!(code(&o), 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn error_classes() {
    assert_eq!(code(&erploc(&[])), 2);
    assert_eq!(code(&erploc(&["resect", "--bogus", "x"])), 2);
    assert_eq!(code(&erploc(&["bogus-subcommand"])), 2);
    assert_eq!(code(&erploc(&["place", "--in", &fixture("place_demo_0.json"), "--seed", "7"])), 2);
    assert_eq!(code(&erploc(&["resect", "--in", "/nonexistent/erploc/none.json"])), 5);
    assert_eq!(code(&erploc(&["site", "--tables", "/nonexistent/erploc/site.bin", "--in", &fixture("place_demo_0.json")])), 3);
    // a text file is not JSON
    let o = erploc(&["resect", "--in", concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")]);
    assert_eq!(code(&o), 3, "{}", String::from_utf8_lossy(&o.stderr));
    // JSON of the wrong subcommand (a place spec given to resect)
    assert_eq!(code(&erploc(&["resect", "--in", &fixture("place_demo_0.json")])), 3);
    // a resect input given to `ba` lacks the BA fields
    assert_eq!(code(&erploc(&["ba", "--in", &fixture("resect_yaw_0.json")])), 3);
    assert_eq!(code(&erploc(&["--help"])), 0);
}

/// Needs the hidden flag: `cargo test -p erploc-cli --features selftest`.
#[cfg(feature = "selftest")]
#[test]
fn panic_is_exit_70() {
    let o = erploc(&["--selftest-panic"]);
    assert_eq!(code(&o), 70);
    assert!(String::from_utf8_lossy(&o.stderr).contains("internal error"));
}
