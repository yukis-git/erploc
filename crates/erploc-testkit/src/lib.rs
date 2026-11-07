//! Dev-only helpers shared by the tests of the other crates.

/// Maximum absolute elementwise difference. Panics on length mismatch.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// True where the fixtures were recorded and compared bit for bit: macOS on aarch64. There the tests demand exact selections, step
/// counts and tight float bounds. Elsewhere (a libm that differs in the last ulp) they compare by objective key within a tolerance.
/// `ERPLOC_TEST_LOOSE=1` selects the off-reference gates on the reference platform too (used to check them with a libm shim).
pub fn strict() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64")) && !matches!(std::env::var("ERPLOC_TEST_LOOSE").as_deref(), Ok("1"))
}

pub mod fixture {
    //! Parser for the plain-text fixtures written by tools/*.py.
    use erploc_linalg::DMat;
    use std::collections::BTreeMap;

    pub struct Fixture {
        pub recs: BTreeMap<(String, String), DMat>,
        pub names: Vec<String>,
    }

    /// Loads `rel` from the workspace's `fixtures/` directory.
    pub fn load(rel: &str) -> Fixture {
        let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut recs = BTreeMap::new();
        let mut names = Vec::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let tag = it.next().unwrap().to_string();
            let name = it.next().unwrap().to_string();
            let r: usize = it.next().unwrap().parse().unwrap();
            let c: usize = it.next().unwrap().parse().unwrap();
            let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
            assert_eq!(v.len(), r * c, "bad record {tag} {name}");
            if !names.contains(&name) {
                names.push(name.clone());
            }
            recs.insert((tag, name), DMat::from_rows(r, c, &v));
        }
        Fixture { recs, names }
    }

    impl Fixture {
        pub fn get(&self, tag: &str, name: &str) -> Option<&DMat> {
            self.recs.get(&(tag.to_string(), name.to_string()))
        }
    }
}

use erploc_linalg::rng::RandomSource;
use erploc_linalg::DMat;

/// Replays recorded numpy draws: a (kind, value) sequence; kind 0 = uniform, 1 = normal.
pub struct Replay {
    draws: Vec<(u8, f64)>,
    pos: usize,
}

impl Replay {
    /// From a fixture record whose 2 columns are (kind, value), one row per draw, in call order.
    pub fn from_record(m: &DMat) -> Replay {
        assert_eq!(m.cols(), 2, "Replay record needs 2 columns (kind, value)");
        let draws = (0..m.rows())
            .map(|i| {
                let k = m[(i, 0)];
                assert!(k == 0.0 || k == 1.0, "Replay: bad kind {k} at row {i}");
                (k as u8, m[(i, 1)])
            })
            .collect();
        Replay { draws, pos: 0 }
    }

    /// Panics unless every draw was consumed (call at the end of a test).
    pub fn assert_exhausted(&self) {
        assert_eq!(self.pos, self.draws.len(), "Replay: {} of {} draws consumed", self.pos, self.draws.len());
    }

    fn next(&mut self, kind: u8) -> f64 {
        let (k, v) = *self.draws.get(self.pos).unwrap_or_else(|| panic!("Replay: draws exhausted at call {}", self.pos));
        assert_eq!(k, kind, "Replay: kind mismatch at draw {} (recorded {k}, requested {kind})", self.pos);
        self.pos += 1;
        v
    }
}

impl RandomSource for Replay {
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        let v = self.next(0);
        assert!(a <= v && v <= b, "Replay: uniform draw {v} outside [{a}, {b}]");
        v
    }
    fn normal(&mut self) -> f64 {
        self.next(1)
    }
}

pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// numpy `std(ddof=1)`.
pub fn std_ddof1(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}

/// numpy `median` (average of the two middle values for even n).
pub fn median(x: &[f64]) -> f64 {
    assert!(!x.is_empty(), "median of empty slice");
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("median: NaN"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_helpers() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!((mean(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-15);
        assert!((std_ddof1(&[1.0, 2.0, 3.0, 4.0]) - (5.0f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn replay_roundtrip() {
        let m = DMat::from_rows(2, 2, &[0.0, 0.5, 1.0, -1.25]);
        let mut r = Replay::from_record(&m);
        assert_eq!(r.uniform(0.0, 1.0), 0.5);
        assert_eq!(r.normal(), -1.25);
        r.assert_exhausted();
    }

    #[test]
    #[should_panic(expected = "kind mismatch")]
    fn replay_kind_mismatch() {
        let m = DMat::from_rows(1, 2, &[1.0, 0.3]);
        Replay::from_record(&m).uniform(0.0, 1.0);
    }
}
