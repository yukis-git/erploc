# Site tables (`ERPSITE1`)

A site table is a precomputed model of one room or yard for the placement search (`erploc site`, `erploc_place::site::SiteTables`).
It holds, for every pair (evaluation cell, candidate marker), whether the marker is visible from the cell and how much Fisher information
the observation adds. The table is the only input the site search needs, so you can produce one with your own geometry code.

`erploc-place` can write a fully synthetic example: `cargo run --release -p erploc-place --example gen_site -- site.bin`
(a 14 m x 10 m room, one box-shaped obstacle, 16 wall markers and 38 free-standing stands; 183,917 bytes). Its geometry lives in
`crates/erploc-place/src/site_synth.rs`.

## Meaning of the entries

- **Cells** `K x (x, y, z)`: camera positions at which the layout is judged. The zone statistics (`erploc site ... zone`) need the cells
  to lie on a 1 m grid in x and y.
- **Candidates** `N`: marker positions `(x, y, z)` with azimuth (direction of the marker normal in the xy plane) and a kind,
  `wall` (flat on a wall, never blocks) or `stand` (a free-standing board on a base, can block the view of other markers).
- **Visibility masks** `K x N` (row-major `k*N + j`): `centre` is the centre-ray test, `samples` is the stricter 13-sample test of the
  marker face. Every samples-visible pair must also be centre-visible.
- **Information**: for each pair that is visible in either mode, the 10 upper-triangle entries of the 4 x 4 Fisher information block of the
  camera state (x, y, z, yaw), in the order (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3). The information is the same in
  both modes.
- **Scalars**: marker side `A`, camera size (`W = 2H`), pixel noise `sigma_px` and its model (0 tangent, 1 pixel), the position accuracy
  target `tau`, the leave-one-out target `tau_loo`, the minimum count `k_min`, the cap `sigma_cap`, the weight `w_k`, the maximum
  incidence angle `psi_max` [rad], the minimum apparent size `px_min` [px], the grid spacing and the camera height.

The loader fixes `n_max = 10`, the eight default headings, `lam_sign = 1` and a single camera height; the workspace polygon is a
placeholder because the cells are given explicitly.

## Byte layout (little endian)

| Field | Type |
|---|---|
| magic | 8 bytes `ERPSITE1` |
| version | u32 = 1 |
| writer tag | u32 length + that many bytes (informational) |
| `A`, then `W`, `H` | f64, u32, u32 |
| `sigma_px`, noise model | f64, u32 |
| `tau`, `tau_loo` | f64, f64 |
| `k_min` | u32 |
| `sigma_cap`, `w_k`, `psi_max`, `px_min` | f64 x 4 |
| `LOCAL_R` (must be 6.0), spacing, camera height | f64 x 3 |
| `K`, `N` | u32, u32 |
| cells | `K` x 3 f64 |
| candidates | `N` x (x y z azimuth as f64, kind u8: 0 wall, 1 stand) |
| centre mask, samples mask | `ceil(K*N/8)` bytes each, LSB first |
| information | u32 `n_pairs`, then `n_pairs` x (u32 k, u32 j, 10 f64) |
| reference block | 10 x u64 |

`n_pairs` must cover every pair visible in either mode. The reference block is, in order: number of centre-visible pairs, number of
blocked triples (centre mode), FNV-1a of the sorted triples (centre mode), the same three numbers for the samples mode, the number of
pairs visible by the centre ray but not by the samples, the number of pairs with a non-zero information entry, FNV-1a of the neighbour lists
and their total length. The loader checks the counts that it can recompute cheaply (visible pairs, non-zero pairs); the triple
checksums are used by the tests. A triple (blocker stand, target candidate, cell) means the stand blocks the view of the target from that
cell.

The loader never panics on a malformed file: every truncation, bad index, non-finite value or inconsistent count is `PlaceError::Site`,
which the CLI reports as exit code 4.

`near` lists (candidates within `LOCAL_R = 6 m` of each other in the xy plane) are derived on load and need not be stored.
