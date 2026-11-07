//! numpy 2.0.2 semantics that the Python placement code depends on, in one place.

pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// numpy `np.mod(x, m)` for f64, m > 0 (numpy `npy_divmod`): `fmod`, then a negative nonzero remainder gets `+ m`
/// (a tiny negative `x` can therefore give exactly `m`), a zero remainder becomes `+0.0`. NaN/inf in -> NaN.
pub fn np_mod(x: f64, m: f64) -> f64 {
    assert!(m > 0.0, "np_mod: modulus must be positive");
    let r = x % m;
    if r != 0.0 {
        if r < 0.0 {
            r + m
        } else {
            r
        }
    } else {
        0.0
    }
}

/// numpy `np.arange` length: `ceil((stop - start)/step)`, 0 if not positive.
/// Panics if `step == 0` or the length is not finite or exceeds `isize::MAX`.
pub fn arange_len(start: f64, stop: f64, step: f64) -> usize {
    assert!(step != 0.0, "arange_len: step must be nonzero");
    let len = ((stop - start) / step).ceil();
    assert!(len.is_finite(), "arange_len: length is not finite");
    if len <= 0.0 {
        return 0;
    }
    assert!(len <= isize::MAX as f64, "arange_len: length exceeds isize::MAX");
    len as usize
}

/// numpy `np.arange(start, stop, step)` values: first element `start`, second `start + step`, then
/// `first + i*delta` with `delta = second - first` (numpy `DOUBLE_fill`). The fill is compiled with FMA on this
/// platform (arm64 numpy 2.0.2): a plain `start + i*delta` differs by 1 ulp (e.g. `arange(0.05, 3.0, 0.1)`), so use `mul_add`.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = arange_len(start, stop, step);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(start);
    if n == 1 {
        return out;
    }
    let second = start + step;
    out.push(second);
    let delta = second - start;
    for i in 2..n {
        out.push((i as f64).mul_add(delta, start));
    }
    out
}

/// `np.round(x)` (half to even) as `usize` for `x >= 0`; panics if `x` is not finite, negative or `>= 2^53`.
pub fn round_half_even_usize(x: f64) -> usize {
    assert!((0.0..9_007_199_254_740_992.0).contains(&x), "round_half_even_usize: out of range");
    x.round_ties_even() as usize
}

/// numpy `pairwise_sum` for a contiguous f64 slice (n < 8 sequential from 0.0; n <= 128 eight accumulators;
/// above that split at n/2 rounded down to a multiple of 8).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum` of a contiguous 1-D f64 array: identity `0.0` plus numpy's pairwise sum.
pub fn np_sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// numpy `min`/`max` over a column: NaN propagates.
pub fn col_min(poly: &[[f64; 2]], c: usize) -> f64 {
    let mut m = poly[0][c];
    for p in &poly[1..] {
        if m.is_nan() {
            break;
        }
        if p[c].is_nan() || p[c] < m {
            m = p[c];
        }
    }
    m
}

pub fn col_max(poly: &[[f64; 2]], c: usize) -> f64 {
    let mut m = poly[0][c];
    for p in &poly[1..] {
        if m.is_nan() {
            break;
        }
        if p[c].is_nan() || p[c] > m {
            m = p[c];
        }
    }
    m
}

/// numpy `a @ b` for 3-vectors, as `(M,3) @ (3,)` computes each row on this platform (arm64 numpy 2.0.2):
/// `((a0*b0 + a1*b1) + a2*b2)`, no fused multiply-add. Measured on random `b` with M from 1 to 5000 (the other associations
/// and the FMA chains differ in 1 to 30 % of the rows) and on 10^6 site rows where `b` is an in-plane axis `(c, s, 0)` as in
/// the occlusion code: there only the FMA chains through `a1*b1` differ (8 to 15 % of the rows), the association is moot.
#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

/// `np.searchsorted(sorted, v, side='left')`: the first index `i` with `sorted[i] >= v` (`sorted.len()` if none).
pub fn searchsorted_left<T: PartialOrd + Copy>(sorted: &[T], v: T) -> usize {
    sorted.partition_point(|&x| x < v)
}

pub(crate) use crate::placement::{nmax, nmin};

/// `np.unique` of integers: sorted ascending, duplicates removed.
pub fn unique_sorted(a: &[i64]) -> Vec<i64> {
    let mut v = a.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// `bincount` into a reusable buffer (cleared first); same values.
pub fn bincount_into(idx: &[usize], weights: &[f64], minlength: usize, out: &mut Vec<f64>) {
    assert_eq!(idx.len(), weights.len(), "bincount: idx/weights length");
    let len = idx.iter().max().map_or(0, |&m| m + 1).max(minlength);
    out.clear();
    out.resize(len, 0.0);
    for (&i, &w) in idx.iter().zip(weights) {
        out[i] += w;
    }
}

/// `np.bincount(idx, weights, minlength)`: the weights are added sequentially in array order (`out[idx[i]] += w[i]`);
/// the length is `max(minlength, max(idx) + 1)`.
pub fn bincount(idx: &[usize], weights: &[f64], minlength: usize) -> Vec<f64> {
    assert_eq!(idx.len(), weights.len(), "bincount: idx/weights length");
    let len = idx.iter().max().map_or(0, |&m| m + 1).max(minlength);
    let mut out = vec![0.0; len];
    for (&i, &w) in idx.iter().zip(weights) {
        out[i] += w;
    }
    out
}

/// `np.einsum('kn,knd->kd', W, Ip)` for ONE row `k` with 0/1 weights and 10-entry blocks: `out[d] = sum_n w[n] * ip[n][d]`.
/// The summation order is the one measured against numpy: a
/// left-to-right loop over `n` starting from `0.0`. With 0/1 weights `w * x` is exact, so an FMA would change nothing.
pub fn sum_weighted10(w: &[bool], ip: &[f64]) -> [f64; 10] {
    assert_eq!(ip.len(), w.len() * 10, "sum_weighted10: block length");
    let mut acc = [0.0; 10];
    for (n, &wn) in w.iter().enumerate() {
        let wf = if wn { 1.0 } else { 0.0 };
        for d in 0..10 {
            acc[d] += wf * ip[n * 10 + d];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUM_IN: [u64; 300] = [
        0x3fa181b6e2396dca,
        0x3f564731cdda4792,
        0x405e9e3703e135ec,
        0xbfe0546f81b70d75,
        0xc007d66a63f0957c,
        0xbfe0e054d0b0306f,
        0x3fad2b846eb02adf,
        0xc0166cffa183ce22,
        0x3f487954a5d8aad6,
        0xbfc7a5505bd58cbd,
        0x402f54bacf5a775d,
        0xbfeedbb8f9b053ee,
        0x3f7bde4900ea1b97,
        0xbf21e667e0572cad,
        0xc042f4762f41590e,
        0x3f3e59b2fc51b954,
        0x40207d825eed0754,
        0xbfc9ec7fb18bb5b8,
        0xbf2406a747cd8341,
        0x3f7c161231558a69,
        0xbf81d3176f7bc165,
        0xbf58cfc8e7d0f2f5,
        0x3fd9476201c16edf,
        0xbf45f91d242ac0fd,
        0xbfc8949271963ffa,
        0xbfb4d6fb6a62286d,
        0xbf3ea4fc391f4baa,
        0xbfbe8bc591ceca47,
        0xbff7e121cf48309a,
        0x3fa2c231790c841d,
        0x3fecb64412c03105,
        0xc037502e62d8de53,
        0xbf7e752b64821624,
        0x3fa3b630db8c04d9,
        0x401cb07ea268cd36,
        0xbf6893ada2de1e20,
        0x3f764f41838d7c61,
        0x3ff0af9e321b9653,
        0xc0008e77c2a0eabc,
        0xbf4aa84352e48287,
        0x3fa1ccb9d63637bb,
        0x3f99594358ae07d7,
        0x3f8680f18861a15e,
        0xbf8a4ee722b91f44,
        0xc0508a52a00631f0,
        0xbfb57502d07c7495,
        0xbffbbe8659f593ba,
        0x3f2092708f2efe4a,
        0x40151cb713265f56,
        0xbfe7a42afdf994ae,
        0x3fc1bc7d03701dbf,
        0x3fea4d344437c72d,
        0x40191855cbf32162,
        0x4010117b09ac68ee,
        0x4057e44903af3c7e,
        0xc060a655e2721ca5,
        0x40188ea3c3691ade,
        0x3faedcb76d5cce0c,
        0xc031ad5c0bf59b78,
        0x3fd635bdbec34547,
        0xbf6483b75e304ca1,
        0x3fb401c982111198,
        0xbf3cc63dbc9fd7aa,
        0xbf27e89dbdb74216,
        0x3f3678180de4333e,
        0xc055e813269d5a43,
        0x404dee097f2b7f9a,
        0xbff0cb4acdbee2d2,
        0x3f742c0dfa0212d3,
        0xc04a178524092284,
        0x3f863ed3086b596e,
        0x40183544aca129e0,
        0xbffc7be7aee6434a,
        0x404f990d365d826e,
        0x402931f248fc0cca,
        0x3f5d58bc101ea2cc,
        0xbf59c80ec31f865c,
        0x3f4cf03f1fde71ae,
        0x404740e07d1b3938,
        0xbf4ec19ce808f9d3,
        0xc0592aa7d457e986,
        0x3fc0178e3efd10c0,
        0xc05f8b1f87492098,
        0x4016ad85858dd973,
        0x3f55546e1de9e007,
        0xbfc479cbdbd47018,
        0xc0083390bc4a9292,
        0xbff4f28d5569b0f8,
        0x3f2ffd18d32f170f,
        0x402e4999c57277a5,
        0x40343c4dade52e9f,
        0xbf5d21f3d85de12f,
        0xbf778cc6fc3af809,
        0x401c244d3fdcb388,
        0x3f59e05e78e458be,
        0x3f3b9abdf12083a4,
        0xbfb319fb52bfed70,
        0x3f68574ca5480005,
        0xbf5b3a991e11be67,
        0xc0004c9c2899a5f1,
        0xbf7e157cd7b2ed9c,
        0x400efb146de29271,
        0x3fd3b44cbf96228b,
        0xbf830b0bd5970b15,
        0xbf2d0e9ce6e08f12,
        0xbf8a50a9d3e26f3c,
        0xbfa8e466d618c085,
        0x40282105b6961357,
        0xbf28fa12f662c6ea,
        0xbf8d7c29075b506c,
        0x402ab058e03cc454,
        0x3f4160304b379926,
        0x4000dd5cbb688a3a,
        0x3f447be69d9e55b6,
        0xbfa79f75bf94c62d,
        0xbf57b7dca8cc49b3,
        0x3f8b1cb9c85e8043,
        0x4039b1f310ade5b4,
        0xc05485fbcd180248,
        0xbf45342a204eee07,
        0x404dcf37749af637,
        0xc054c235f61d159b,
        0xbfd150d40a78ffe6,
        0xbf6ca8a0d8f6b51c,
        0x3fc8921bdb8c479d,
        0x3f866c0070c1d03f,
        0x3f6213e8b397d2bf,
        0x3fed67b6252ae0a8,
        0xbf3b84911386cd4f,
        0x3f6ada6efcfeaa41,
        0xc06aba5267265e4c,
        0xbfc28f309862df9a,
        0x3fe9781f67bd4fa4,
        0xbf4356882f2f6dd0,
        0x401732542d4f18a4,
        0x3fabc49d57853e02,
        0x3ff5280de0451c03,
        0x3f4a9a5e0e1d58ce,
        0x3f84d3f6af2476d7,
        0xbf524bd5c862e408,
        0xbf46e1a3ff91375a,
        0xc05249f9e6c3d625,
        0xbfdf3c1ca6a344f5,
        0xbf87238ef79494cd,
        0xbf766c5cad0ed425,
        0xbf4e55413e87605b,
        0x3f649caf68a81a27,
        0xc00b1c7bbed539f3,
        0xbffec767af6dce60,
        0xc01ce9c648535349,
        0x40020714eb7ddcd0,
        0x3f8635c680fc0411,
        0x3f77ab5795503f00,
        0xbfb079a3a9ffa61c,
        0xc01cf378d365e1b4,
        0x40341b213138393e,
        0x3fb35eb3d6bddbf7,
        0x4032507c66f882ba,
        0x4001091a36e1e02d,
        0xbfea2dbdcc2ec822,
        0x400ed250b832c102,
        0x3f72c43ab04d0fce,
        0x3fe1e857e5193438,
        0x3f763241b5a2739b,
        0x3f2a7bd41d75ef40,
        0x3ffbdc5576d4c9e2,
        0xbf589df106fdd474,
        0xbffa76d51c4f5e50,
        0xbf7e9f5c5a83b609,
        0x402937ab883412e4,
        0xbf3ea41fe5768c9f,
        0x4018bda709b5b00d,
        0x40547a15a48a956c,
        0x3f9f9c2f64e50574,
        0x403f9de78cf35fba,
        0x3f185d8be3767f7c,
        0xc04663d022ba84d6,
        0xc030733b7c3b0a61,
        0xbf744d39c52ad11d,
        0x400f09f9fc61fed6,
        0x3fc210aecb7b3394,
        0x3fe299efffef48ee,
        0xbf55c4c142f4ee93,
        0x4021c16d98d78977,
        0xbf48fdf5b2706f38,
        0xbfb2cc31159b4b48,
        0xbf602cadf4190cd6,
        0xbfacd864fbe0f14a,
        0x403d1cce14caad2a,
        0xbfe25f8ecf2dd70c,
        0xbfbb60ed96541867,
        0xbfb5a70da415d506,
        0x402a3d5ac8ba6d09,
        0x3fdc5348d92e8964,
        0xc02759c14b8a8880,
        0x3feacec4c867c590,
        0xc06373074d8c9a31,
        0x40665e9cec9e5453,
        0xbff86443361ea7eb,
        0x3fa8831b81dd66ff,
        0x3fabd56909e63067,
        0xbfc28f62aa15a6af,
        0x3f9ebf919a8b99d8,
        0x4058ede6abacf03a,
        0x3fddecc0e430f622,
        0x3f9aba62905d5d7c,
        0x4022fb60009857c8,
        0x3ff9bea67b88b5c6,
        0x3f360c479c1e5d77,
        0xbf209bbbe577125e,
        0x3fb02c9a1d10f284,
        0xbfb81974848925d7,
        0x3f49f1caa6c9fcd0,
        0xbfe02b6e1f47b099,
        0xbff17350f3891841,
        0x400d384c71fb19c7,
        0x3ffb16588f7602d6,
        0x40580abd94f57316,
        0xbfaa66b9b525f6e4,
        0x3f46dfbfbb1f3acf,
        0xc0122b19f46deddc,
        0xbfbfbfbd342c5551,
        0x3f8255df18578d67,
        0xc00289a9b45a44da,
        0x3f5f64e7e29f81ca,
        0xbf50de7e72cae80a,
        0xbf7c90c125ee395b,
        0xc0626e0c9fdcc4c7,
        0xc02e54c7300c5c0c,
        0xbfee2f38d3128c54,
        0x3f80e881eea34be9,
        0x4064d35512643f74,
        0xc0042a0154d6e791,
        0x3f865c20fc4c4317,
        0xc0051e52da99ab48,
        0xbffe9770d11535d6,
        0x402dfbbb11cbce78,
        0x3f7240f5f283bff0,
        0xbfa5f1f6810aa66f,
        0x3fd357a450803395,
        0x3fe252617ad347c4,
        0xbfeba108e8436158,
        0xbf8e3edd2c0d712e,
        0xbfcc520e761fb540,
        0xc0351b265b51bd0e,
        0xbf6ce3e2366752aa,
        0x3f843804294f22d3,
        0x3ffb9c02fd8db96b,
        0xbfa55513673e4765,
        0x3fb1e439515e8c46,
        0x405785f589114075,
        0x4051d4337aae4570,
        0x3f512979d628a423,
        0xbfa3ede6084a86b4,
        0x3fe7ac7e62ef70a7,
        0x3fc9e9d2894affad,
        0x40541775a70978f8,
        0xbf79640e65996d85,
        0x404e62b84cd4f92b,
        0x3f89da8225706f9e,
        0x404273064363cc14,
        0xc04c2b8766afdc36,
        0x3f8f9a2fb322351b,
        0xbf899860ef689397,
        0x3f407f090e4184d8,
        0xbff9e183ead08629,
        0xbf538e98d020a954,
        0x3ff382ddabe1b375,
        0x3f6c7fbb4c0ac3fb,
        0xc05d2b822223d2ac,
        0x3fe3415bd3904944,
        0xbf3c490dba2420ea,
        0xc033072379082057,
        0xc05138fe284efd46,
        0x3fd0de06e3f54e3d,
        0x40196d98f965267a,
        0x3ffa7d89dc0e9a8c,
        0x3fa71fde4ca3943f,
        0x4012ccb29ab515b0,
        0xc000e0ca62ae40a0,
        0x3f881aa1ba9fec87,
        0x3ff7c85ca6300979,
        0x3fd2cf9c57ddb569,
        0x3fabb05c4c7ad105,
        0xbff0ea29dfb753e2,
        0xbfe6b1fc8febb9fa,
        0x3f590b5353773025,
        0x3f6b77eed2eecbf1,
        0xc0069bfa2fae1398,
        0x3f366124969f49d7,
        0xbf3f9047bfa548af,
        0x400649099f938111,
        0xbf7b462a00b9c43e,
        0x3f9c43aaed6bc14d,
        0xc02f8dd64a14847a,
        0x4060a2f17eb66a27,
        0xc01484b0492a5f32,
        0xbf90312101209600,
        0xbf62564cf27974aa,
        0xc00dae00878469b4,
    ];
    const SUM_OUT: [(usize, u64); 8] = [
        (3, 0x405ea07d81ef4af4),
        (8, 0x405c3c361ca090f0),
        (9, 0x405c3c42594ae3dc),
        (13, 0x405fddbf1238eb02),
        (100, 0xc064122057899db1),
        (129, 0xc069feba0f7c09e5),
        (200, 0xc0743736bffdf7ba),
        (300, 0x4052c6837453bcc0),
    ];

    #[test]
    fn np_sum_matches_numpy() {
        let a: Vec<f64> = SUM_IN.iter().map(|&b| f64::from_bits(b)).collect();
        for &(n, want) in &SUM_OUT {
            assert_eq!(np_sum(&a[..n]).to_bits(), want, "n = {n}");
        }
    }

    type AraCase = (f64, f64, f64, usize, &'static [u64]);
    const ARA: [AraCase; 6] = [
        (
            0.25,
            8.0,
            0.5,
            16,
            &[
                0x3fd0000000000000,
                0x3fe8000000000000,
                0x3ff4000000000000,
                0x3ffc000000000000,
                0x4002000000000000,
                0x4006000000000000,
                0x400a000000000000,
                0x400e000000000000,
                0x4011000000000000,
                0x4013000000000000,
                0x4015000000000000,
                0x4017000000000000,
                0x4019000000000000,
                0x401b000000000000,
                0x401d000000000000,
                0x401f000000000000,
            ],
        ),
        (
            0.05,
            3.0,
            0.1,
            30,
            &[
                0x3fa999999999999a,
                0x3fc3333333333334,
                0x3fd0000000000001,
                0x3fd6666666666668,
                0x3fdcccccccccccce,
                0x3fe199999999999a,
                0x3fe4ccccccccccce,
                0x3fe8000000000001,
                0x3feb333333333335,
                0x3fee666666666668,
                0x3ff0ccccccccccce,
                0x3ff2666666666667,
                0x3ff4000000000001,
                0x3ff599999999999b,
                0x3ff7333333333334,
                0x3ff8ccccccccccce,
                0x3ffa666666666668,
                0x3ffc000000000002,
                0x3ffd99999999999b,
                0x3fff333333333335,
                0x4000666666666667,
                0x4001333333333334,
                0x4002000000000001,
                0x4002ccccccccccce,
                0x400399999999999b,
                0x4004666666666668,
                0x4005333333333334,
                0x4006000000000001,
                0x4006ccccccccccce,
                0x400799999999999b,
            ],
        ),
        (0.1, 1.0, 0.3, 3, &[0x3fb999999999999a, 0x3fd999999999999a, 0x3fe6666666666667]),
        (
            0.35,
            8.0,
            0.1,
            77,
            &[
                0x3fd6666666666666,
                0x3fdccccccccccccc,
                0x3fe1999999999999,
                0x3fe4cccccccccccc,
                0x3fe7ffffffffffff,
                0x3feb333333333332,
                0x3fee666666666665,
                0x3ff0cccccccccccc,
                0x3ff2666666666666,
                0x3ff3ffffffffffff,
                0x3ff5999999999998,
                0x3ff7333333333332,
                0x3ff8cccccccccccc,
                0x3ffa666666666665,
                0x3ffbfffffffffffe,
                0x3ffd999999999998,
                0x3fff333333333332,
                0x4000666666666666,
                0x4001333333333332,
                0x4001ffffffffffff,
                0x4002cccccccccccc,
                0x4003999999999998,
                0x4004666666666665,
                0x4005333333333332,
                0x4005ffffffffffff,
                0x4006cccccccccccc,
                0x4007999999999998,
                0x4008666666666665,
                0x4009333333333332,
                0x4009fffffffffffe,
                0x400acccccccccccb,
                0x400b999999999998,
                0x400c666666666665,
                0x400d333333333332,
                0x400dfffffffffffe,
                0x400ecccccccccccb,
                0x400f999999999998,
                0x4010333333333332,
                0x4010999999999999,
                0x4010ffffffffffff,
                0x4011666666666665,
                0x4011cccccccccccc,
                0x4012333333333332,
                0x4012999999999998,
                0x4012ffffffffffff,
                0x4013666666666665,
                0x4013cccccccccccc,
                0x4014333333333332,
                0x4014999999999998,
                0x4014ffffffffffff,
                0x4015666666666665,
                0x4015cccccccccccc,
                0x4016333333333332,
                0x4016999999999998,
                0x4016ffffffffffff,
                0x4017666666666665,
                0x4017cccccccccccb,
                0x4018333333333332,
                0x4018999999999998,
                0x4018fffffffffffe,
                0x4019666666666665,
                0x4019cccccccccccb,
                0x401a333333333332,
                0x401a999999999998,
                0x401afffffffffffe,
                0x401b666666666665,
                0x401bcccccccccccb,
                0x401c333333333332,
                0x401c999999999998,
                0x401cfffffffffffe,
                0x401d666666666665,
                0x401dcccccccccccb,
                0x401e333333333331,
                0x401e999999999998,
                0x401efffffffffffe,
                0x401f666666666664,
                0x401fcccccccccccb,
            ],
        ),
        (
            -1.3,
            2.9,
            0.7,
            7,
            &[
                0xbff4cccccccccccd,
                0xbfe3333333333334,
                0x3fb9999999999990,
                0x3fe9999999999998,
                0x3ff7ffffffffffff,
                0x4001999999999999,
                0x4007333333333332,
            ],
        ),
        (
            0.0,
            1.0,
            0.1,
            10,
            &[
                0x0000000000000000,
                0x3fb999999999999a,
                0x3fc999999999999a,
                0x3fd3333333333334,
                0x3fd999999999999a,
                0x3fe0000000000000,
                0x3fe3333333333334,
                0x3fe6666666666667,
                0x3fe999999999999a,
                0x3feccccccccccccd,
            ],
        ),
    ];

    #[test]
    fn arange_matches_numpy() {
        for &(s, e, st, n, bits) in &ARA {
            let v = arange(s, e, st);
            assert_eq!(v.len(), n);
            assert_eq!(arange_len(s, e, st), n);
            for (x, &b) in v.iter().zip(bits) {
                assert_eq!(x.to_bits(), b, "arange({s}, {e}, {st})");
            }
        }
    }

    #[test]
    fn arange_len_formula() {
        assert_eq!(arange_len(0.25, 8.0, 0.5), 16);
        for &(s, e, st) in &[(0.1_f64, 1.0, 0.3), (0.05, 3.0, 0.1), (0.0, 1.0, 0.1), (5.0, 1.0, 1.0), (0.0, 0.0, 1.0)] {
            let want = ((e - s) / st).ceil().max(0.0) as usize;
            assert_eq!(arange_len(s, e, st), want);
        }
        assert!(arange(1.0, 0.0, 1.0).is_empty());
    }

    #[test]
    #[should_panic]
    fn arange_zero_step_panics() {
        arange_len(0.0, 1.0, 0.0);
    }

    #[test]
    fn np_mod_cases() {
        assert_eq!(np_mod(5.5, 2.0), 1.5);
        assert_eq!(np_mod(-0.5, 2.0), 1.5);
        assert_eq!(np_mod(-1e-20, TWO_PI), TWO_PI);
        let z = np_mod(-0.0, 2.0);
        assert!(z == 0.0 && z.is_sign_positive());
        let z = np_mod(-4.0, 2.0);
        assert!(z == 0.0 && z.is_sign_positive());
        assert_eq!(np_mod(-1e18, 3.0), 3.0 - (1e18 % 3.0));
        assert!(np_mod(f64::NAN, 2.0).is_nan());
        assert!(np_mod(f64::INFINITY, 2.0).is_nan());
        assert!(np_mod(f64::NEG_INFINITY, 2.0).is_nan());
    }

    #[test]
    fn round_half_even() {
        assert_eq!(round_half_even_usize(0.5), 0);
        assert_eq!(round_half_even_usize(1.5), 2);
        assert_eq!(round_half_even_usize(2.5), 2);
        assert_eq!(round_half_even_usize(2.6), 3);
        assert_eq!(round_half_even_usize(0.0), 0);
    }

    #[test]
    fn col_min_max_propagate_nan() {
        let p = [[1.0, 5.0], [f64::NAN, 2.0], [0.5, 9.0]];
        assert!(col_min(&p, 0).is_nan() && col_max(&p, 0).is_nan());
        assert_eq!(col_min(&p, 1), 2.0);
        assert_eq!(col_max(&p, 1), 9.0);
        let q = [[f64::NAN, 0.0], [1.0, 1.0]];
        assert!(col_min(&q, 0).is_nan() && col_max(&q, 0).is_nan());
        let r = [[3.0, 0.0], [1.0, 1.0], [2.0, 7.0]];
        assert_eq!((col_min(&r, 0), col_max(&r, 0)), (1.0, 3.0));
    }

    #[test]
    #[should_panic]
    fn round_half_even_negative_panics() {
        round_half_even_usize(-1.0);
    }

    #[test]
    #[should_panic]
    fn round_half_even_too_large_panics() {
        round_half_even_usize(9_007_199_254_740_992.0);
    }

    #[test]
    #[should_panic]
    fn round_half_even_nan_panics() {
        round_half_even_usize(f64::NAN);
    }

    #[test]
    #[should_panic]
    fn arange_len_infinite_panics() {
        arange_len(0.0, f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn arange_len_too_long_panics() {
        arange_len(0.0, 1e300, 1.0);
    }

    #[test]
    fn dot3_association() {
        // 1e16 + 1 rounds back to 1e16: the left-to-right association is visible
        let a = [1e16, 1.0, -1e16];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(dot3(&a, &b), 0.0); // (1e16 + 1) rounds to 1e16, then - 1e16 = 0
        let c = [1.0, 1e16, -1e16];
        assert_eq!(dot3(&c, &b), 0.0);
        assert_eq!(dot3(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn searchsorted_left_generic_i64() {
        let s = [-5i64, 0, 0, 7, 100];
        assert_eq!(searchsorted_left(&s, 0), 1);
        assert_eq!(searchsorted_left(&s, 7), 3);
        assert_eq!(searchsorted_left(&s, 8), 4);
        assert_eq!(searchsorted_left(&s, -9), 0);
    }

    #[test]
    fn unique_sorted_semantics() {
        assert_eq!(unique_sorted(&[5, 1, 5, 3, 1, 9]), vec![1, 3, 5, 9]);
        assert!(unique_sorted(&[]).is_empty());
        assert_eq!(unique_sorted(&[-2, -2]), vec![-2]);
    }

    #[test]
    fn bincount_adds_sequentially() {
        // 1e16 + 1 + 1 differs by order: sequential gives 1e16 (each +1 is absorbed), summing the 1s first gives 1e16 + 2
        let w = [1e16, 1.0, 1.0, 4.0];
        let b = bincount(&[2, 2, 2, 0], &w, 4);
        assert_eq!(b, vec![4.0, 0.0, 1e16, 0.0]);
        assert_eq!(bincount(&[], &[], 3), vec![0.0; 3]);
        assert_eq!(bincount(&[5], &[2.0], 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn sum_weighted10_is_left_to_right_from_zero() {
        // column 0: 1e16, 1, 1 -> sequential 1e16; column 1: -0.0 entries sum to +0.0 (0.0 + -0.0)
        let mut ip = vec![0.0; 30];
        ip[0] = 1e16;
        ip[10] = 1.0;
        ip[20] = 1.0;
        ip[1] = -0.0;
        ip[11] = -0.0;
        let s = sum_weighted10(&[true, true, true], &ip);
        assert_eq!(s[0], 1e16);
        assert!(s[1] == 0.0 && s[1].is_sign_positive());
        // zero weight drops the block
        let s = sum_weighted10(&[true, false, true], &ip);
        assert_eq!(s[0], 1e16);
        assert_eq!(sum_weighted10(&[], &[]), [0.0; 10]);
    }

    #[test]
    fn searchsorted_left_semantics() {
        let s = [0u32, 0, 1, 3, 3, 3, 7];
        assert_eq!(searchsorted_left(&s, 0), 0);
        assert_eq!(searchsorted_left(&s, 1), 2);
        assert_eq!(searchsorted_left(&s, 2), 3);
        assert_eq!(searchsorted_left(&s, 3), 3);
        assert_eq!(searchsorted_left(&s, 4), 6);
        assert_eq!(searchsorted_left(&s, 8), 7);
        assert_eq!(searchsorted_left(&[], 5), 0);
    }
}
