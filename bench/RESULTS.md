# erploc benchmark results

Produced by `python bench/bench.py` (about 6 minutes; `--quick` runs a reduced version). Everything below is measured on synthetic rooms
generated from the seed; no number comes from a real site or camera. The numbers will differ on another machine; the counts and ratios
in part B are deterministic for a given seed (the random baselines use the seeded generator) and should reproduce exactly on the same
platform.

- date: 2026-11-08 01:08; seed 20261108; python 3.12.14, numpy 2.5.4, erploc 0.1.0
- machine: Apple M4, 10 logical CPUs, macOS-27.2-arm64-arm-64bit
- load average before: 3.17 2.86 3.06 (1/5/15 min); the machine was shared with other work, so the timings are upper bounds
- conditions: placement tau=0.03 m, k_min=2, tau_loo=0.1 m, tag side 0.16 m, 6528x3264 camera, 4 headings, cells every 0.7 m, 2 m wall spacing

## A. Time

Single process, release build, one thread. Estimators on an 8 m x 6 m room with 14 markers, 4096 px wide frames, 0.3 px noise.

| operation | median | min | max |
|---|---|---|---|
| resection, yaw (4 dof), 14 markers | 0.13 ms | 0.11 ms | 0.15 ms |
| resection, full (6 dof), 14 markers | 0.18 ms | 0.17 ms | 0.20 ms |
| resection, full, Huber loss | 0.25 ms | 0.25 ms | 0.26 ms |
| fisher_camera, 14 markers | 0.02 ms | 0.02 ms | 0.02 ms |
| init_bundle, 10 frames, 14 markers | 3.56 ms | 3.54 ms | 3.58 ms |
| bundle_adjust (incl. init), 10 frames, 14 markers | 59.78 ms | 59.28 ms | 60.69 ms |
| init_bundle, 40 frames, 14 markers | 13.91 ms | 13.79 ms | 14.06 ms |
| bundle_adjust (incl. init), 40 frames, 14 markers | 363.00 ms | 361.69 ms | 364.36 ms |

Placement search on random synthetic rooms (the search is deterministic; times include building the score tables on the first call).

| scene | operation | median | spread |
|---|---|---|---|
| small room: 12 candidates, 50 cells | evaluate (tables cached) | 0.55 ms | |
| | place_markers greedy -> 6 markers | 106.13 ms | min 105.11 ms, max 106.30 ms |
| | place_markers ift -> 6 markers | 25.46 ms | min 24.95 ms, max 25.52 ms |
| large room: 16 candidates, 88 cells | evaluate (tables cached) | 0.99 ms | |
| | place_markers greedy -> 7 markers | 551.41 ms | min 545.15 ms, max 551.96 ms |
| | place_markers ift -> 6 markers | 52.07 ms | min 51.06 ms, max 52.09 ms |

## B. Placement quality on synthetic rooms

Small rooms (5-8 m x 4-6 m, 0-2 box obstacles, candidates every 2 m, 200 random draws per count, exhaustive enumeration up to 12000 subsets):

| room | cand. | cells | greedy count (time) | IFT count (time) | random, same count: mean feasible / any full | evenly spaced: feasible | smallest count of 200 random draws | exhaustive min count |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 34 | 4 (0.04 s) | 4 (0.01 s) | 84% / 3% | 91% | 4 | 4 |
| 2 | 14 | 63 | 3 (0.06 s) | 3 (0.01 s) | 82% / 0% | 83% | 4 | 3 |
| 3 | 12 | 38 | 5 (0.07 s) | 5 (0.02 s) | 92% / 7% | 95% | 5 | 5 |
| 4 | 12 | 42 | 3 (0.04 s) | 3 (0.01 s) | 88% / 2% | 90% | 3 | 3 |
| 5 | 12 | 38 | 5 (0.14 s) | 5 (0.01 s) | 91% / 6% | 100% | 5 | 5 |
| 6 | 12 | 42 | 6 (0.09 s) | 6 (0.02 s) | 88% / 2% | 98% | 6 | 6 |
| 7 | 12 | 44 | 5 (0.15 s) | 5 (0.02 s) | 93% / 1% | 91% | 5 | 5 |
| 8 | 10 | 36 | 5 (0.07 s) | 5 (0.01 s) | 90% / 4% | 94% | 5 | 5 |
| 9 | 12 | 32 | 5 (0.07 s) | 4 (0.01 s) | 80% / 0% | 94% | 4 | 4 |
| 10 | 10 | 35 | 4 (0.05 s) | 3 (0.01 s) | 90% / 10% | 94% | 3 | 3 |
| 11 | 14 | 60 | 4 (0.13 s) | 4 (0.02 s) | 95% / 9% | 98% | 4 | 4 |
| 12 | 12 | 38 | 6 (0.07 s) | 5 (0.01 s) | 84% / 2% | 95% | 5 | 5 |
| 13 | 10 | 34 | 6 (0.07 s) | 5 (0.01 s) | 91% / 0% | 97% | 5 | 5 |
| 14 | 10 | 36 | 4 (0.04 s) | 4 (0.01 s) | 87% / 4% | 94% | 4 | 4 |
| 15 | 10 | 28 | 5 (0.05 s) | 5 (0.01 s) | 91% / 12% | 96% | 5 | 5 |
| 16 | 12 | 41 | 5 (0.08 s) | 5 (0.02 s) | 87% / 2% | 93% | 5 | 5 |
| 17 | 12 | 32 | 3 (0.07 s) | 3 (0.01 s) | 89% / 14% | 97% | 3 | 3 |
| 18 | 10 | 35 | 3 (0.02 s) | 3 (0.01 s) | 92% / 7% | 91% | 3 | 3 |
| 19 | 10 | 29 | 6 (0.04 s) | 5 (0.01 s) | 83% / 1% | 93% | 5 | 5 |
| 20 | 10 | 31 | 4 (0.03 s) | 4 (0.01 s) | 92% / 13% | 97% | 4 | 4 |
| 21 | 12 | 44 | 5 (0.14 s) | 5 (0.02 s) | 89% / 2% | 89% | 5 | 5 |
| 22 | 14 | 55 | 5 (0.10 s) | 5 (0.02 s) | 93% / 2% | 91% | 5 | 5 |
| 23 | 12 | 49 | 3 (0.10 s) | 3 (0.01 s) | 89% / 4% | 86% | 3 | 3 |
| 24 | 12 | 36 | 3 (0.04 s) | 3 (0.01 s) | 93% / 8% | 89% | 3 | 3 |

Rooms: 24. Exhaustive enumeration was run on 24 of them (those with at most the stated number of subsets up to the IFT count).
- IFT count equals the exhaustive minimum count in 24 of 24; greedy in 19 of 24.
- Worst-case sigma_pos of the IFT layout over the best layout of the same size (exhaustive min-max), where the counts agree: median 1.17, max 4.11 (n=24).
- The same for greedy: median 1.27, max 8.89 (n=19).
- Markers used, summed over the rooms: IFT 102, greedy 107, random layouts (smallest count at which any of the draws was fully feasible; all candidates if none was) 103.
- Mean feasible fraction of equally many random markers: 89%; of evenly spaced ones: 93%.

Larger rooms (8-11 m x 6-8 m, candidates every 1.5 m, no exhaustive enumeration):

| room | cand. | cells | greedy count (time) | IFT count (time) | random, same count: mean feasible / any full | evenly spaced: feasible | smallest count of 200 random draws | exhaustive min count |
|---|---|---|---|---|---|---|---|---|
| 1 | 22 | 101 | 8 (1.14 s) | 7 (0.11 s) | 92% / 0% | 92% | 8 | not run |
| 2 | 20 | 68 | 5 (0.36 s) | 5 (0.03 s) | 90% / 2% | 97% | 5 | not run |
| 3 | 22 | 104 | 6 (1.01 s) | 5 (0.05 s) | 87% / 0% | 94% | 6 | not run |
| 4 | 24 | 112 | 6 (0.59 s) | 4 (0.05 s) | 84% / 0% | 98% | 5 | not run |
| 5 | 22 | 108 | 4 (0.60 s) | 4 (0.05 s) | 89% / 0% | 98% | 4 | not run |
| 6 | 24 | 109 | 7 (1.70 s) | 7 (0.13 s) | 92% / 0% | 96% | 7 | not run |
| 7 | 22 | 108 | 5 (0.67 s) | 4 (0.05 s) | 87% / 0% | 95% | 5 | not run |
| 8 | 24 | 100 | 6 (0.53 s) | 6 (0.09 s) | 93% / 0% | 96% | 7 | not run |

Rooms: 8.
- Markers used, summed over the rooms: IFT 42, greedy 47, random layouts (smallest count at which any of the draws was fully feasible; all candidates if none was) 47.
- Mean feasible fraction of equally many random markers: 89%; of evenly spaced ones: 96%.

- load average after: 3.31 2.96 3.08

## Reading part B

- The IFT search minimises, in this order, the number of infeasible cells, then a penalty, then (negated) margin. It does not minimise the
  worst-case sigma_pos: a layout that is feasible everywhere can have a worst-case sigma_pos several times that of the best layout of the same size
  (the ratio column of the summary: median 1.17, maximum 4.11 for IFT over the 24 small rooms, maximum 8.89 for greedy).
- On the 24 small rooms (10 to 14 candidates) the IFT count equals the exhaustive minimum in all 24. That is a check on small instances where
  enumeration is possible; it says nothing about rooms with hundreds of candidates, where it cannot be checked this way.
- The baselines are weak by design (random layouts, evenly spaced layouts). With 10 to 14 candidates the best of 200 random draws is already
  close to IFT, because there are only a few thousand subsets. On the 8 larger rooms (20 to 24 candidates) the smallest count at which a random draw
  was fully feasible was the same as IFT's in 3 rooms and one marker more in the other 5, and at IFT's own count only 2 % of random draws in
  one room were fully feasible (0 % in the rest). Greedy needed more markers than IFT in 4 of the 8 rooms (47 against 42 in total).
- "Feasible" is the model's feasibility (sigma_pos below 30 mm, at least 2 visible markers, and 100 mm with any one removed), computed from the
  Fisher information and a visibility model; no image was rendered or detected for these runs.

## C. IFT neighbourhood evaluation (Rust, synthetic site)

`cargo run --release -p erploc-place --example bench_nbhd`: one `FastEval::neighbourhood` call (the keys of all single swaps of a nine-marker
selection, 200 calls after 20 warm-up calls) on the synthetic site of the tests (92 cells, 54 candidates). Apple M4, release build, one
thread, load average about 2.4 to 3.7 (machine shared with other work).

| visibility mode | median | min | max |
|---|---|---|---|
| Centre | 0.897 ms | 0.851 ms | 1.088 ms |
| Samples | 0.888 ms | 0.823 ms | 1.072 ms |

The design target of 5 ms per evaluation is met on this site. The cost grows with cells times candidates; scenes with hundreds of candidates were
not measured here, so the target is not claimed for them.
