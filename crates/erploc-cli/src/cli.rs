//! Hand-written argument parser, file I/O and the `catch_unwind` net of the `erploc` binary.
use crate::cmd_ba::run_ba;
use crate::cmd_place::{run_place, Method, PlaceOpts};
use crate::cmd_resect::run_resect;
use crate::cmd_site::run_site;
use crate::e2e::{format_table, run_e2e, E2eConsts, PY_IFT_BUDGET_S};
use crate::error::CliError;
use erploc_place::search::Budget;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const USAGE: &str = "\
usage:
  erploc place  --in spec.json [--out result.json] [--csv report.csv] [--method greedy|ift] [--budget-ms N] [--sigma-ts 0.005]
  erploc resect --in obs.json [--out result.json]
  erploc ba     --in problem.json [--out result.json]
  erploc site   --tables site.bin --in query.json [--out result.json]
  erploc e2e    [--seed-base N] [--budget-s S]
`--in -` reads standard input; without `--out` the result goes to standard output.
`place` is deterministic (there is no --seed); --budget-ms applies to --method ift (default: no wall-clock limit).
exit codes: 0 ok, 1 an e2e check failed, 2 usage, 3 invalid input, 4 numerical failure, 5 i/o error, 70 internal error";

/// `--key value` pairs of one subcommand, checked against the allowed key list.
struct Flags {
    pairs: Vec<(String, String)>,
}

impl Flags {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Flags, CliError> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        let mut it = args.iter();
        while let Some(k) = it.next() {
            if !allowed.contains(&k.as_str()) {
                return Err(CliError::Usage(format!("unknown or misplaced argument {k:?}\n{USAGE}")));
            }
            let Some(v) = it.next() else {
                return Err(CliError::Usage(format!("{k} needs a value")));
            };
            if pairs.iter().any(|(pk, _)| pk == k) {
                return Err(CliError::Usage(format!("{k} given twice")));
            }
            pairs.push((k.clone(), v.clone()));
        }
        Ok(Flags { pairs })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn need(&self, key: &str) -> Result<&str, CliError> {
        self.get(key).ok_or_else(|| CliError::Usage(format!("{key} is required\n{USAGE}")))
    }

    fn num<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, CliError> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v.parse::<T>().map(Some).map_err(|_| CliError::Usage(format!("{key}: cannot parse {v:?}"))),
        }
    }
}

fn read_input(path: &str) -> Result<String, CliError> {
    let bytes = if path == "-" {
        let mut b = Vec::new();
        std::io::stdin().read_to_end(&mut b).map_err(|e| CliError::Io(format!("stdin: {e}")))?;
        b
    } else {
        std::fs::read(path).map_err(|e| CliError::Io(format!("{path}: {e}")))?
    };
    String::from_utf8(bytes).map_err(|_| CliError::Input(format!("{path}: not valid UTF-8")))
}

fn write_output(path: Option<&str>, text: &str) -> Result<(), CliError> {
    match path {
        Some(p) => std::fs::write(p, text).map_err(|e| CliError::Io(format!("{p}: {e}"))),
        None => {
            let mut so = std::io::stdout().lock();
            match so.write_all(text.as_bytes()).and_then(|_| so.write_all(b"\n")) {
                Ok(()) => Ok(()),
                // `erploc ... | head`: the reader closed the pipe; not an error of ours
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                Err(e) => Err(CliError::Io(format!("stdout: {e}"))),
            }
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), CliError> {
    #[cfg(any(test, feature = "selftest"))]
    if args.first().map(String::as_str) == Some("--selftest-panic") {
        panic!("selftest panic");
    }
    let Some(sub) = args.first() else {
        return Err(CliError::Usage(USAGE.into()));
    };
    let rest = &args[1..];
    match sub.as_str() {
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        "place" => {
            let f = Flags::parse(rest, &["--in", "--out", "--csv", "--method", "--budget-ms", "--seed", "--sigma-ts"])?;
            let method = match f.get("--method") {
                None | Some("greedy") => Method::Greedy,
                Some("ift") => Method::Ift,
                Some(m) => return Err(CliError::Usage(format!("--method must be greedy or ift, got {m:?}"))),
            };
            if f.get("--seed").is_some() {
                return Err(CliError::Usage("--seed is not supported: place is deterministic".into()));
            }
            let opts = PlaceOpts { method, budget_ms: f.num("--budget-ms")?, sigma_ts: f.num("--sigma-ts")?.unwrap_or(0.005) };
            let out = run_place(&read_input(f.need("--in")?)?, &opts)?;
            if let Some(csv) = f.get("--csv") {
                write_output_file(Path::new(csv), &out.csv)?;
            }
            write_output(f.get("--out"), &out.json)
        }
        "resect" => {
            let f = Flags::parse(rest, &["--in", "--out"])?;
            let out = run_resect(&read_input(f.need("--in")?)?)?;
            write_output(f.get("--out"), &out)
        }
        "ba" => {
            let f = Flags::parse(rest, &["--in", "--out"])?;
            let out = run_ba(&read_input(f.need("--in")?)?)?;
            write_output(f.get("--out"), &out)
        }
        "site" => {
            let f = Flags::parse(rest, &["--tables", "--in", "--out"])?;
            let tables = PathBuf::from(f.need("--tables")?);
            let out = run_site(&read_input(f.need("--in")?)?, &tables)?;
            write_output(f.get("--out"), &out)
        }
        "e2e" => {
            let f = Flags::parse(rest, &["--seed-base", "--budget-s"])?;
            let mut c = E2eConsts::default();
            if let Some(s) = f.num::<u64>("--seed-base")? {
                c.seed = s;
            }
            let budget_s = f.num::<f64>("--budget-s")?.unwrap_or(PY_IFT_BUDGET_S);
            if !(budget_s.is_finite() && budget_s > 0.0) {
                return Err(CliError::Usage("--budget-s must be a positive number".into()));
            }
            if let Err(e) = crate::e2e::check_seed_base(&c) {
                return Err(CliError::Usage(format!("--seed-base: {e}")));
            }
            let lines = run_e2e(&c, &Budget::Wall(budget_s))?;
            print!("{}", format_table(&lines));
            let failed: Vec<&str> = lines.iter().filter(|l| !l.passed).map(|l| l.name.as_str()).collect();
            if failed.is_empty() {
                Ok(())
            } else {
                Err(CliError::CheckFailed(failed.join("; ")))
            }
        }
        other => Err(CliError::Usage(format!("unknown subcommand {other:?}\n{USAGE}"))),
    }
}

fn write_output_file(path: &Path, text: &str) -> Result<(), CliError> {
    std::fs::write(path, text).map_err(|e| CliError::Io(format!("{}: {e}", path.display())))
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".into()
    }
}

/// Runs the CLI and returns the exit code. A panic anywhere below is a bug by the CLI error model; it is caught here, reported and
/// mapped to exit code 70 instead of the default abort-like 101.
pub fn run(args: &[String]) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatch(args))) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("erploc: {e}");
            e.exit_code()
        }
        Err(p) => {
            eprintln!("erploc: internal error (this is a bug): {}", panic_message(p.as_ref()));
            CliError::Internal(String::new()).exit_code()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn usage_errors_exit_2() {
        assert_eq!(run(&s(&[])), 2);
        assert_eq!(run(&s(&["bogus"])), 2);
        assert_eq!(run(&s(&["resect"])), 2);
        assert_eq!(run(&s(&["resect", "--in"])), 2);
        assert_eq!(run(&s(&["resect", "--in", "a", "--in", "b"])), 2);
        assert_eq!(run(&s(&["resect", "--nope", "a"])), 2);
        assert_eq!(run(&s(&["place", "--in", "a", "--method", "sa"])), 2);
        assert_eq!(run(&s(&["place", "--in", "a", "--budget-ms", "x"])), 2);
        assert_eq!(run(&s(&["place", "--in", "a", "--seed", "7"])), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file system access is not available under Miri isolation")]
    fn missing_file_exits_5() {
        assert_eq!(run(&s(&["resect", "--in", "/nonexistent/erploc/none.json"])), 5);
    }

    /// The catch_unwind net turns a panic into exit code 70 (flag compiled only under `cfg(test)` / feature `selftest`).
    #[test]
    fn panic_exits_70() {
        assert_eq!(run(&s(&["--selftest-panic"])), 70);
    }
}
