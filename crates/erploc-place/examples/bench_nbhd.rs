//! Time of one `FastEval::neighbourhood` call (all single-swap keys of a nine-marker selection) on the synthetic site.
//!
//! `cargo run --release -p erploc-place --example bench_nbhd`
use erploc_place::fast_eval::{FastEval, IftContext};
use erploc_place::site::VisMode;
use erploc_place::site_synth::synthetic_site;
use erploc_place::zone::Crit;
use std::time::Instant;

fn main() {
    let site = synthetic_site().expect("synthetic site");
    let (k, n) = (site.cells.len(), site.cands.len());
    let sel: Vec<usize> = (0..9).map(|i| i * n / 9).collect();
    println!("synthetic site: {k} cells, {n} candidates, selection {sel:?}");
    for mode in [VisMode::Centre, VisMode::Samples] {
        let mut ctx = IftContext::from_site(&site, mode, Crit::N);
        let mut ev = FastEval::new(&mut ctx, sel.len()).expect("fast evaluator");
        ev.set_key(&sel).expect("set_key");
        for _ in 0..20 {
            ev.neighbourhood(&sel).expect("neighbourhood");
        }
        let mut t: Vec<f64> = (0..200)
            .map(|_| {
                let t0 = Instant::now();
                let nb = ev.neighbourhood(&sel).expect("neighbourhood");
                std::hint::black_box(&nb);
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{mode:?}: median {:.3} ms, min {:.3} ms, max {:.3} ms (200 calls, nine markers)", t[100], t[0], t[199]);
    }
}
