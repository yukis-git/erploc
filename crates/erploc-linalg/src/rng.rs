//! PCG64 (XSL-RR 128/64) for tests and synthetic data. Not stream-compatible with numpy (by design, see spec §3).

const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    spare: Option<f64>,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 53-bit mantissa from a u64: uniform on [0, 1).
pub fn u64_to_unit(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let st = ((splitmix64(&mut s) as u128) << 64) | splitmix64(&mut s) as u128;
        let inc = ((((splitmix64(&mut s) as u128) << 64) | splitmix64(&mut s) as u128) << 1) | 1;
        let mut r = Pcg64 { state: 0, inc, spare: None };
        r.step();
        r.state = r.state.wrapping_add(st);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.step();
        let rot = (old >> 122) as u32;
        let xored = ((old >> 64) as u64) ^ (old as u64);
        xored.rotate_right(rot)
    }

    pub fn uniform01(&mut self) -> f64 {
        u64_to_unit(self.next_u64())
    }

    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform01()
    }

    /// Standard normal, Marsaglia polar method (caches the second value).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform01() - 1.0;
            let v = 2.0 * self.uniform01() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }
}

/// Draw interface shared by `Pcg64` and the test-only replay source (numpy `Generator` semantics, one draw per call).
pub trait RandomSource {
    /// numpy `Generator.uniform(a, b)` (one draw).
    fn uniform(&mut self, a: f64, b: f64) -> f64;
    /// numpy `Generator.normal()` (one standard-normal draw; callers scale it: `loc + scale * z`).
    fn normal(&mut self) -> f64;
}

impl RandomSource for Pcg64 {
    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        Pcg64::uniform(self, a, b)
    }
    fn normal(&mut self) -> f64 {
        Pcg64::normal(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Pcg64::new(7);
        let mut b = Pcg64::new(7);
        for _ in 0..1000 {
            let x = a.uniform01();
            assert_eq!(x, b.uniform01());
            assert!((0.0..1.0).contains(&x));
        }
        assert_ne!(Pcg64::new(1).next_u64(), Pcg64::new(2).next_u64());
    }

    #[test]
    fn normal_moments() {
        let mut r = Pcg64::new(20261004);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "var {var}");
    }
}
