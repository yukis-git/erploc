use erploc_linalg::{eig, lu, svd};
use erploc_testkit::fixture;

fn rel(a: &[f64], b: &[f64], scale: f64) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale.max(f64::MIN_POSITIVE)
}

#[test]
fn square_solve_inv_det_match_numpy() {
    let f = fixture::load("linalg.txt");
    let mut n_cases = 0;
    for name in f.names.iter().filter(|n| n.starts_with("sq")) {
        let a = f.get("A", name).unwrap();
        let b = f.get("b", name).unwrap().as_slice().to_vec();
        let x = lu::solve(a, &b).unwrap();
        let xs = f.get("solve", name).unwrap().as_slice();
        assert!(rel(&x, xs, 1.0) < 1e-12, "{name} solve");
        let ai = lu::inv(a).unwrap();
        let an = f.get("inv", name).unwrap();
        assert!(rel(ai.as_slice(), an.as_slice(), an.max_abs()) < 1e-12, "{name} inv");
        let d = f.get("det", name).unwrap()[(0, 0)];
        assert!((lu::det(a) - d).abs() <= 1e-12 * d.abs().max(1.0), "{name} det");
        n_cases += 1;
    }
    assert!(n_cases >= 18);
}

#[test]
fn eigvalsh_matches_numpy_relative_to_norm() {
    let f = fixture::load("linalg.txt");
    let mut n_cases = 0;
    for name in f.names.iter().filter(|n| n.starts_with("sym")) {
        let a = f.get("S", name).unwrap();
        let w = eig::eigvalsh(a).unwrap();
        let wn = f.get("eigvalsh", name).unwrap().as_slice();
        let scale = wn.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let n = a.rows() as f64;
        assert!(rel(&w, wn, scale) < 4.0 * n * f64::EPSILON * 10.0, "{name}: {w:?} vs {wn:?}");
        n_cases += 1;
    }
    assert!(n_cases >= 25);
}

#[test]
fn singular_values_and_pinv_match_numpy() {
    let f = fixture::load("linalg.txt");
    let (mut n_cases, mut worst) = (0, 0.0f64);
    for name in f.names.iter().filter(|n| n.starts_with("rect")) {
        let a = f.get("M", name).unwrap();
        let fs = svd::svd(a).unwrap();
        let sn = f.get("sv", name).unwrap().as_slice();
        assert!(rel(&fs.s, sn, sn[0]) < 1e-13, "{name} sv");
        let p = svd::pinv(a, svd::PINV_RCOND).unwrap();
        let pn = f.get("pinv", name).unwrap();
        let e = rel(p.as_slice(), pn.as_slice(), pn.max_abs());
        worst = worst.max(e);
        assert!(e < 1e-11, "{name} pinv");
        n_cases += 1;
    }
    println!("pinv: worst rel err {worst:e}");
    assert!(n_cases >= 7);
}

/// Indices whose value is separated from both neighbours (sorted list) by more than `gap · scale`.
fn separated(vals: &[f64], scale: f64, gap: f64) -> Vec<usize> {
    (0..vals.len())
        .filter(|&i| {
            let lo = i == 0 || (vals[i] - vals[i - 1]).abs() > gap * scale;
            let hi = i + 1 == vals.len() || (vals[i + 1] - vals[i]).abs() > gap * scale;
            lo && hi
        })
        .collect()
}

fn abs_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>().abs()
}

#[test]
fn eigenvectors_match_numpy_up_to_sign() {
    let f = fixture::load("linalg.txt");
    let (mut compared, mut worst) = (0, 0.0f64);
    for name in f.names.iter().filter(|n| n.starts_with("sym")) {
        let a = f.get("S", name).unwrap();
        let e = eig::eigh(a).unwrap();
        let wn = f.get("eigvalsh", name).unwrap().as_slice();
        let vn = f.get("eigvecs", name).unwrap();
        let scale = wn.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for k in separated(wn, scale, 1e-6) {
            let d = abs_dot(&e.vectors.column(k), &vn.column(k));
            worst = worst.max(1.0 - d);
            assert!(d >= 1.0 - 1e-10, "{name} eigenvector {k}: |dot| = {d}");
            compared += 1;
        }
    }
    println!("eigenvectors: compared {compared}, worst 1-|dot| {worst:e}");
    assert!(compared > 0);
}

#[test]
fn singular_vectors_match_numpy_up_to_sign() {
    let f = fixture::load("linalg.txt");
    let (mut compared, mut worst) = (0, 0.0f64);
    for name in f.names.iter().filter(|n| n.starts_with("rect")) {
        let a = f.get("M", name).unwrap();
        let fs = svd::svd(a).unwrap();
        let sn = f.get("sv", name).unwrap().as_slice();
        let (un, vtn) = (f.get("U", name).unwrap(), f.get("Vt", name).unwrap());
        let row = |m: &erploc_linalg::DMat, k: usize| (0..m.cols()).map(|j| m[(k, j)]).collect::<Vec<f64>>();
        for k in separated(sn, sn[0], 1e-6) {
            let du = abs_dot(&fs.u.column(k), &un.column(k));
            let dv = abs_dot(&row(&fs.vt, k), &row(vtn, k));
            worst = worst.max(1.0 - du).max(1.0 - dv);
            assert!(du >= 1.0 - 1e-10 && dv >= 1.0 - 1e-10, "{name} triplet {k}: |u.u'| = {du}, |v.v'| = {dv}");
            compared += 1;
        }
        if (a.rows(), a.cols()) == (8, 9) {
            let d = abs_dot(&row(&fs.vt, 8), &row(vtn, 8));
            worst = worst.max(1.0 - d);
            assert!(d >= 1.0 - 1e-10, "{name} null vector: |dot| = {d}");
            compared += 1;
        }
    }
    println!("singular vectors: compared {compared} (incl. 8x9 null vector), worst 1-|dot| {worst:e}");
    assert!(compared > 0);
}

#[test]
fn lstsq_matches_numpy() {
    let f = fixture::load("linalg.txt");
    let (mut n_cases, mut worst) = (0, 0.0f64);
    for name in f.names.iter().filter(|n| n.starts_with("rect")) {
        let a = f.get("M", name).unwrap();
        let b = f.get("lb", name).unwrap().as_slice().to_vec();
        let x = svd::lstsq(a, &b).unwrap();
        let xn = f.get("lstsq", name).unwrap().as_slice();
        let scale = xn.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let e = rel(&x, xn, scale);
        worst = worst.max(e);
        assert!(e <= 1e-10, "{name} lstsq rel err {e:e}");
        n_cases += 1;
    }
    println!("lstsq: {n_cases} cases, worst rel err {worst:e}");
    assert!(n_cases >= 7);
}

#[test]
fn large_psd_100_rank_95() {
    let f = fixture::load("linalg.txt");
    let a = f.get("S", "big0").unwrap();
    assert_eq!((a.rows(), a.cols()), (100, 100));
    let w = eig::eigvalsh(a).unwrap();
    let wn = f.get("eigvalsh", "big0").unwrap().as_slice();
    let scale = wn.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let n = a.rows() as f64;
    let e = rel(&w, wn, scale);
    let null = w.iter().filter(|x| x.abs() <= 1e-9 * scale).count();
    println!("large PSD: eigvalsh rel err {e:e} (tol {:e}), null count {null}", 4.0 * n * f64::EPSILON * 10.0);
    assert!(e < 4.0 * n * f64::EPSILON * 10.0, "eigvalsh rel err {e:e}");
    assert_eq!(null, 5);
}
