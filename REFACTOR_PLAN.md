# rmc-discord refactor plan

Working notes for the rewrite. Lives in this repo; no shared core with
discord-tools or discordant-parsimony.

## Goals

- Single-crystal RMC for occupational, displacive, and magnetic disorder,
  with rigid groups (molecules) as a first-class scatterer.
- No rebinning: the model is evaluated at sampled Q points taken from the
  MDHisto data as measured.
- Exact displacive scattering (no Taylor expansion of exp(iQ.u)).
- Mantid for I/O (LoadMD/SaveMD) and crystallography (CrystalStructure,
  SpaceGroup, PointGroup, UnitCell, ReflectionGenerator).
- qtpy + pyvistaqt GUI with a left control panel and a 3D viewer per tab:
  supercell colored by atom, data/calc volumes, 3D correlation maps.
- Library usable from scripts (dataclass state, small functions); public API
  documented later.

## Why the old engine is replaced

- Q is mapped to the nearest supercell k point (JAC 2021, Fig. 2), which
  forces commensurate rebinning of the data.
- Every move blurs the full grid (`filters.filtering`) and recomputes chi^2.
- Displacements use a Taylor expansion with (p+1)(p+2)(p+3)/6 terms that
  fails for Q.u >~ 1 (e.g. 1.5 A CO3 shifts at Q ~ 10 1/A).
- Kernels take ~75 positional arguments threaded through presenter -> model
  -> Cython; molecule variants duplicate every kernel and are unused.
- Unused in the GUI workflow: `diffuse/scattering.py`, `interaction.py`,
  `simulation.pyx`, `structural.py`; `material/structure.py` duplicates the
  GUI pipeline.
- Python 3.12+/NumPy 2 breakage: `distutils.strtobool`, `np.bool = bool`.

## Core: direct sampled-Q update

Structure factor kept at the sample points Q_s. A move of group g changes

    dF(Q) = P_g(Q) * [ c' exp(iQ.r') - c exp(iQ.r) ]

- P_g(Q): group prefactor. Single atom: b_j or f_j(Q). Rigid group:
  sum_a b_a exp(iQ.d_a), precomputed per orientation variant; continuous
  rotations recompute it (O(n_Q n_atoms_in_group)).
- c: occupancy (0/1), or moment vector (projected perpendicular to Q at
  intensity time).
- Occupational, displacive, and magnetic moves share one kernel; combined
  occupancy + displacement moves are just a different proposal.
- Cost per move O(n_Q). F recomputed from scratch every few sweeps to bound
  roundoff drift.
- chi^2 kept incrementally via running weighted sums; scale/background (and
  moment sizes, below) solved in closed form.

### Sampling (replaces rebinning)

1. Default: commensurate points K + m/N inside the measured, unmasked volume;
   observed value from the MD bin containing each point. Any data grid works.
2. Minibatch: random subset of (1) per sweep, redrawn periodically; held-out
   points give an overfitting check.
3. Resolution: average I_calc over a few jittered Q inside the bin or
   resolution ellipsoid, instead of grid blurring.

Note: a periodic supercell carries N^3 independent k per zone; off-grid Q add
no information, only the supercell shape function.

### Masks

- Bragg punch from ReflectionGenerator (allowed reflections for the space
  group) with box/ellipsoid radius and the existing IQR outlier option.
- Arbitrary masks (powder rings, detector gaps) as boolean MD workspaces.

## Moment and displacement size

Bragg peaks are masked, so the absolute scale is lost (JAC 2021, Sec. 6).
Zone sum rules show which sizes the scale absorbs. The Brillouin-zone
average of diffuse intensity per site is

    occupational:  c(1-c) b^2                    (fixed by composition)
    displacive:    b^2 [1 - exp(-Q.U.Q)]         (fixed by <u u^T> = U)
    magnetic:      (2/3) C mu^2 f(Q)^2           (fixed by |mu|)

independent of the correlations. Occupational size is therefore never
ambiguous; moment and displacement sizes trade off against the scale factor.
Correlations (the shape) are determined either way.

Two modes:

- Relative scale (default). Sizes taken from outside: mu_eff per species
  (g sqrt(S(S+1)) or from Curie-Weiss), U_j from the CIF. Scale and
  background float.
- Absolute scale. Scale fixed from integrated nuclear Bragg intensities
  against |F_calc|^2 of the CIF (garnet-tools integration) or from
  normalization to absolute units. Sizes then refine.

### Magnetic

- Spins are unit vectors in the kernel; moment sizes live in the prefactor.
- Keep F per magnetic species s: F = sum_s mu_s F_s(Q), so
  I = mu^T A(Q) mu with A_ss' = Re(F_s,perp . F_s',perp^*). The few mu_s
  solve in closed form with the scale (relative mode: ratios only).
- Optional free-length spins for dilute/spin-glass cases.
- Proposals: rotation within an adaptive cone (target acceptance), not a
  fresh uniform direction every move as now.
- Temperature-difference data (e.g. bixbyite 7 K - 300 K): model the
  subtracted reference explicitly,
  I_model = s [I_calc - (2/3) C mu^2 f^2(Q) w_ref] + bkg, so the high-T
  paramagnetic term is not absorbed into the correlations.

### Displacive

- Current behavior: each candidate is a fresh draw from N(0, U_j). With no
  proposal ratio in the acceptance, this is an implicit Gaussian prior
  N(0, U_j), but its weight vanishes as T -> 0, so sizes drift late in the
  anneal.
- New: explicit, non-annealed constraint on the site second moment,
  cost = chi^2 / T + lambda sum_j || L_j^-1 Cov_j(u) L_j^-T - I ||^2,
  with L_j = chol(U_j), plus <u>_j = 0. Covariance sums update in O(1) per
  move.
- U_j as a target (energy-integrated data, e.g. TOPAZ) or an upper bound
  (elastic data, e.g. CORELLI, where phonon TDS is partly removed).
- Rigid groups: the same constraint on center-of-mass translation (TLS T
  tensor); rotation amplitude bounded similarly when rotations are enabled.
- Proposals: local steps u' = u + delta L_j z with adaptive delta.

## Package layout

    src/rmc_discord/
      crystal/     Mantid CrystalStructure wrapper, Supercell, tables
      data/        LoadMD/SaveMD, QSample (sampling, masks, resolution)
      model/       scatterers and group prefactors, disorder state
      engine/      numba kernels, Metropolis loop, annealing, cost/scale
      analysis/    pair correlations (kD tree), recalculation, symmetrize
      gui/         qtpy MVP, one model/view/presenter per tab, viewers
    tests/

Old `disorder/` stays until the new path reproduces both paper cases, then is
removed. Build via pyproject + versioningit, Python >= 3.11.

## Compute backend

- Numba (`prange` over Q) as the reference backend.
- Narrow backend interface (propose / delta_F / accept / recompute) so a GPU
  backend can be added after benchmarking. GPU only pays off with many moves
  per launch (CuPy RawKernel loop or JAX fori_loop). RTX 2000 Ada FP64 is
  1/64 rate, so float32/complex64 plus periodic recomputation.

## GUI

Tabs: Crystal | Data | Refinement | Correlations | Recalculation.

- Supercell viewer: spheres colored by species (radii/colors tables), vacancies
  hidden or translucent, moments as arrows, displacements as arrows with an
  exaggeration slider or per-site ellipsoids of Cov_j(u). Merged glyph meshes
  for ~1e5-1e6 atoms; clip box and per-species toggles.
- Data viewer: volume or orthogonal slices of data, calc, and residual in hkl,
  non-orthogonal axes handled with the UB/B matrix.
- Correlation viewer: points at separation vectors r colored by C_jj'(r)
  (diverging map), size by |C| or significance; filter by pair type;
  symmetrize with the Laue/point group. "Map onto structure": pick an origin
  atom, draw neighbors colored by correlation with bond tubes.
- Refinement runs in a QThread and emits snapshots (chi^2, acceptance,
  temperature, scale, sizes, current slice).

## Phases

1. Engine without GUI: dataclasses, numba kernel for all disorder types,
   Mantid I/O and crystallography, sampling and masks, scale/size solvers.
   Check against brute-force F(Q) on small supercells and the old test
   reference values.
2. Regression cases: bixbyite (magnetic, temperature difference) and
   Ba3Co2O6(CO3)0.7 (occupational + rigid CO3 displacive). Data stay outside
   the repo: `$DIFFUSE_DATA_ROOT` (same convention as discordant-parsimony;
   CO3 is in ~/git/discordant-parsimony/local_data/Ba3Co2O6_CO3_0p7_50K,
   bixbyite 6/240/300 K volumes in local_data/bixbyite_CORELLI, from
   analysis.sns.gov:/SNS/EXAMPLES/CORELLI/IPTS-12345/shared).
3. Correlations and recalculation on the new state.
4. GUI.
5. Optional GPU backend.

## Phase 1 status (2026-11-03)

Done, in `src/rmc_discord/` (tests in `tests/`, 33 passing; run with
`~/miniforge3/envs/discord/bin/python -m pytest tests`):

- `crystal/`: CIF reader (PyCifRW), `Crystal` on Mantid `UnitCell` /
  `SpaceGroup` (Cartesian ADPs rotated per operator, co-located occupants
  grouped into mixed sites), tables.
- `data/`: `LoadMD`/`SaveMD` volumes (W from `W_MATRIX`), commensurate
  sampling (`mean` combines every bin by inverse variance; integer hkl
  dropped since supercell diffuse vanishes there), Bragg and IQR masks.
- `model/`: `Supercell` with kinds (site, occupant incl. vacancy), unit spins,
  displacements; nuclear/X-ray/magnetic prefactors; paramagnetic offset.
- `engine/`: Numba exact incremental dF/dF_M with candidate chi^2 sums in one
  pass; closed-form scale/background; whitened covariance penalty; swap,
  cone-rotation, and ADP-shaped displacement moves; step/cone adaptation;
  automatic starting temperature.

Verified: kernels vs brute force; per-site Parseval sum rule; drift < 1e-10
after mixed moves; recovery of a noisy AFM (staggered moment, scale) and of
anticorrelated displacements at Q.u ~ 1.5 with <u u^T> held at U.

Findings:

- Free least-squares scale collapses to 0 from a random start when chi^2
  changes per move are << T (they scale as 1/n atoms). Fixed by
  `calibrate_temperature()` (greedy trial moves, then T from median uphill
  delta). Proportional (2% of y) weights also drive collapse; real counting
  errors do not.
- Bixbyite 6 K - 300 K with paramagnetic offset: N=6, ROI 0-4,0-4,0-2, 8k
  points, chi^2_red 695 -> 34 in 17 sweeps, scale stable ~3.5e-8.
- 300 K data is nearly flat in Q (not f^2-like) in that ROI, so it does not
  calibrate the absolute scale cleanly; moment size remains tied to scale.
- Speedups done (2026-11-03): bixbyite N=10 (32k atoms, 35k points) went
  from ~1.0-1.4 ms to ~0.12 ms per spin move (sweep 44 s -> 4 s), ~0.22 ms
  per displacement move; full recompute 1.4 s -> 0.2 s. What mattered, in
  order:
  1. Recompute was being run at the end of every `run()` call; it is now
     opt-in (`recompute_every`), since drift is ~1e-14 after 1e4 moves.
  2. Thread launch overhead: TBB 40-90 us per parallel region on this WSL
     laptop vs OpenMP 5-20 us; engine prefers OpenMP and defaults threads to
     physical cores (all 32 hyperthreads with OpenMP spin-wait = 10x slower).
  3. Specialized evaluate/apply kernels per move type with tuples unpacked
     outside `prange` (passing tuples into inlined helpers inside the loop was
     ~4x slower).
  4. No sin/cos at average positions: lattice phases are N-th roots of unity
     via per-move tables (no integer division per point), site phases
     tabulated; trig only for displaced atoms. Numba here has no SVML.
  5. dF computed again on accept instead of buffered; int16 m instead of
     Cartesian Q; factored prefactors c_k T_t(Q).
  6. Recompute by per-site FFTs for undisplaced atoms, direct sum for
     displaced ones.
  Minibatch (`run(..., subset=f)`) implemented. float32 not done: per-move
  delta chi^2 is ~1e-6 relative, too close to float32 precision.

Rigid groups and combined moves (2026-11-03):

- `Supercell(..., groups={"C1": ["O7"], "C2": ["O8"]})`: each member atom
  joins its nearest center (minimum image, `cutoff` 1.6 A). A group kind has
  `members` (label, b, Cartesian offset d, U); its prefactor is the table
  sum_a b_a exp(iQ.d_a) (per-member Debye-Waller factor on non-displacive
  sites), so kernels and the FFT recompute are unchanged. Group ADP target =
  mean member U (translation only; no rotations yet). `Supercell.atoms()`
  expands groups for brute-force checks and the future 3D view.
- Displacement statistics are per kind, not per site: vacancies excluded,
  each occupant against its own ADP. Displacements stay with positions, so a
  swap moves each displacement into its new kind's sums (O(1) update).
- New move `swap_displace` (weight key `"swap_displace"`): swap on a
  mixed displacive site that also steps the incoming occupants'
  displacements (exact two-atom dF with four phases).
- `Crystal.with_occupancies`; `Volume / Volume` (norm exact);
  `commensurate(counts, ..., norm=norm)` aggregates sum(d)/sum(n) (keeps
  zero-count bins, no Poisson low bias); `powder_ring_mask`; `load()` no
  longer NaNs zero-variance bins (norm files have zero errors).
- Synthetic size effect (groups beside a vacancy along c shift toward it by
  0.15 A; data averaged over 8 truth configurations, since one truth of the
  same N carries its exact speckle): recovered ~80-90% of
  <(v_up - v_down) u_z>, correct sign, ~0 for the null case. In tests.
- CO3 data are raw counts (variance = counts) plus norm; the lattice from
  the data (c = 9.599 A) differs from the CIF (9.518 A). Carbonate
  occupancies scaled x 0.7/0.6. N = (10,10,16) gives 5.5M points to
  |Q| = 15.5 1/A; first runs use 100k random points with |Q| < 10.

- Radial background (`background_basis="radial"`, `knot_spacing`): linear
  hat functions in |Q| fitted in closed form with the scale. The basis is
  eliminated analytically (P = W - W B G^-1 B^T W), so each candidate needs
  only B^T W x (K sums), reduced by chunks from a per-point buffer that the
  evaluate kernels fill. `background_curve()` returns it.
- `calibrate_temperature()` with a free scale clamped at 0 (random start
  uncorrelated with data) used to give T ~ 0 (CO3: 3.5e-5, swaps frozen from
  the start). It now calibrates at the fluctuation-matched scale
  sqrt(y'Py / x'Px) with the scale fixed, then frees it.
- Numba 0.68 bug: in a prange loop that calls an inlined tuple-returning
  helper, stores into an array unpacked from a tuple argument are silently
  dropped. Arrays written in such loops must be separate arguments.

Not yet: group rotations (orientation variants as kinds), species-resolved
moment fit, resolution jitter.

Displacement exchange (2026-11-13): new move `exchange` (weight key
`"exchange"`, default 0, `kernels.MOVE_EXCHANGE = 4`) swaps the displacements
of two atoms on the same displacive site via the two-atom pair kernels.
Swaps keep displacements with positions, so swap + exchange (with
`displace`/`swap_displace` off) conserve each site's set of displacements
exactly: start from `randomize_displacements()` and the distribution stays
at the ADP while only its arrangement refines. Motivation: on CO3 the
penalized displace moves let rms u grow to 1.3-2.3x the ADP, adding
uncorrelated Q^2<u^2> scattering that held-out off-sheet points reject.
36 tests pass.

CO3 regression recipe (2026-11-14, accepted as good): carbonate-only rigid groups C1/O7, C2/O8 (no Ba moves), N = (8, 8, 20),
occupancies scaled 0.7/0.6, data lattice; commensurate counts/norm sampling with Bragg (0.15, 0.15, 0.3) and powder-ring masks,
2.5 <= |Q| < 10 1/Å; variance += (0.03 y)^2 (counting errors alone are far too small); radial background
(0.5 1/Å knots), symmetrize ["k,h,l"] (data 6/mmm, model 6/m); moves swap + exchange (displacements drawn once from N(0, U),
so their distribution stays at the ADP), penalty 10; scale fixed at max(free, 0.25 x matched) during the anneal; T0 = 0.01,
x0.92 per round of 4 sweeps, 60 rounds; minibatch ~100k points per 2 sweeps from a 619k pool (all on-sheet points + 3x off-sheet);
50k + 50k held-out points scored every 5 rounds, best state kept. Result: held-out on-sheet chi2 36.3 -> 31.3, off-sheet 16.6 -> 17.6;
planes l = 0.8 and 2.85 reproduce the data's cluster lattice. Lessons: fixed point sets overfit (held-out rises while training
falls); unconstrained displacements inflate rms u to 1.3-2.3x the ADP; plot slices in the volume's own W projection with the
metric aspect.

Bixbyite check (2026-11-14): 6 K - 300 K difference, full volume (7.8M commensurate points at N = 10 after Bragg mask;
the old [0,4]x[0,4]x[0,2] box gave only 35k points for 32k spins), variance += (0.03 y)^2, magnetic only with
paramagnetic offset, spin moves (cone adapts to 180), calibrated T0 = 1.8e-4, x0.9 per 2 sweeps, 40 rounds, 100k-point
minibatches from a 7.0M pool, 777k held out. Held-out chi2: background only 5.889, random spins 5.889, refined 5.646 (-4.1%),
refined spins shuffled within sites 5.889. First two Fe-Fe shells (3.0-3.1, 3.6-3.7 A) <S.S> = -0.044, -0.047 (weak
short-range AFM). Not yet compared with the old code's reference values.

## Phase 3 status (2026-11-14)

`analysis/correlations.py`: `correlations(supercell, quantity, states=None, symmetrize=True, normalize=False,
kinds=None)` for "occupancy" (delta = [kind = k] - c_k on mixed sites), "spin" (S on magnetic kinds), and
"displacement" (u on occupied atoms; scalar u.u plus Cartesian tensor). No k-d tree: atoms sit at R + x_j, so
C_jj'(dR) = sum_R a_j(R).a_j'(R+dR) / sum_R m_j m_j' is a cyclic cross-correlation over the cell grid, done by FFT
for every site pair and every separation at once (exact, no cutoff; masks give the pair counts). Symmetrization
averages over the crystal's own space-group operators as index permutations (j, j', dR) -> (sigma j, sigma j',
W dR + t_j' - t_j), rotating displacement tensors; sums and counts are symmetrized before dividing; the supercell
must be invariant under W. Several states give mean and standard error. `Correlations.shells()` averages by site
pair and distance. Timing: CO3 8x8x20 (3 sites) 0.01 s, bixbyite N=10 (32 sites, 48 ops, 1M pair vectors) 3.4 s.
Tests (44 total): FFT = brute-force pair sums for all quantities; symmetrized correlations invariant when the
configuration is moved by each P-6 operator (raw ones are not); pair-reversal identity; AFM pattern; normalization.
Next: run several independent refinements per case for error bars; correlation export (and later the GUI viewer).

Export, script API, packaging (2026-11-14):
- `Correlations.save/load` (HDF5, format "rmc_discord.correlations/1"), `pair(a, b)`, `within(r_max)`;
  `write_shells/read_shells` (CSV: label, distance, value, error, n).
- `Refinement.score(sample, fit="background", scale=None)`: held-out chi^2 with the same settings and state.
- `engine/anneal.py`: `split_sample` (stratified held-out sets), `Schedule`, `anneal` (minibatches, geometric
  cooling, calibrated or given T0, optional scale floor, optional step/cone adaptation, held-out checks, best
  state kept/saved/restored), returning `AnnealResult`. `examples/co3.py`, `examples/bixbyite.py` package the
  two regression recipes (data under $DIFFUSE_DATA_ROOT).
- Packaging: one build for both packages. pyproject.toml holds metadata, dependencies, versioningit
  (default-tag 0.0.4, writes src/rmc_discord/_version.py, gitignored) and finds rmc_discord in src/ and
  disorder in .; setup.py only defines the legacy Cython extensions. Verified up to compiling the Cython
  extensions (isolated pip wheel: version 0.0.4.post381+gd6a62c0..., both packages and data collected);
  this machine has no gcc, so the full build was not completed here.

Seed results (2026-11-14, 4 independent runs each, fixed data split): held-out chi2 CO3 on 31.3-31.7 / off 17.4-17.7
(background only 36.3 / 16.6); bixbyite 5.641-5.652 (background and shuffled 5.889). Shell correlations reproducible:
CO3 normalized occupancy C1-C2 3.00 A +0.206(4), 6.59 A +0.260(9), C2-C2 3.59 A +0.196(21), along c 9.60 A +0.10-0.15,
next channel 9.68 A -0.01 to -0.02; displacement similar. Bixbyite <S.S> nn -0.036 to -0.054 (+-1-2e-4), 4.70 A
+0.061-0.067. Errors are run-to-run only (not data treatment).

Recalculation (2026-11-14): `analysis/recalculation.py`, `recalculate(supercell, corr, hkl, r_max=None)` for
occupancy (binary sites, contrast P_k - P_k') and spins (needs the spin tensor <S_a S_b>, now computed by
`correlations`; perpendicular projection exact). Numba sum over pair vectors per Q. Exact at commensurate points
without truncation (tests: occupancy and spin equal Refinement.intensity() to 1e-10); symmetrized correlations give
the average over space-group-equivalent Q. Off the grid, truncate at r_max <= `max_range(supercell)` (inscribed
sphere radius, min_i N_i d_i / 2): beyond it minimum-image vectors stand for periodic images and are not closed
under the group. `Correlations.normalized` flag (recalculation refuses normalized input). 49 tests.
Bixbyite: 4-run averaged spin correlations, r < 20 A, on the data's (h,k,0) and (h,k,0.5) bin centres (3 s per
40k-point plane), paramagnetic term subtracted, one scale + background: Pearson r 0.32 (l = 0) and 0.67 (l = 0.5)
against the 6 K - 300 K data; the diffuse streak pattern is reproduced. Not yet: displacement recalculation
(pair <u u> gives only the first-order term), mixed-species moments (site prefactor = fraction-weighted mean).

Displacement recalculation (2026-11-14): `recalculate` on displacement correlations gives the whole nuclear diffuse
intensity with vacancies in the Gaussian pair approximation, I = (1/n) sum P_j* P_j' D_j D_j' e^{iQ.r}
[n_jj' exp(Q.C_jj'.Q) - N_c c_j c_j'] (C over occupied pairs, D from the on-site tensor, last term removes Bragg;
all orders in u, not first-order). Tests: exact for zero displacements (= Refinement.intensity); for a 400-member
ensemble of correlated Gaussian displacements around fixed vacancies, mean |z| 0.65, mean z 0.02 against the
ensemble mean (dropping inter-atomic correlations: mean |z| 3.2). Size-effect cross terms <delta u> not included.
CO3: 4-run averaged correlations, r < 33.5 A (max_range of 8x8x20), data bin centres (2.5 <= |Q| < 10), one scale
plus quadratic-in-|Q| background: Pearson r 0.70 (l = 2.82, cluster lattice reproduced) and 0.38 (l = 0.78). 51 tests.

## Speed (2026-11-15)

CO3 run-13 settings (3840 atoms, 100k-observation minibatch x 2 symmetry equivalents = 200k points,
radial basis, swap + exchange). Profile before: 3.9 ms/move (evaluate_pair alone 2-4 ms, memory-bound:
~150 B/point read incl. unused FM, Phi gather, tiled weights, xbuf, then separate obs_sums/basis_sums passes),
recompute of the displaced atoms 2.5 s per minibatch (one sincos per atom per point), resample(1.0) 16 s per
round. ~85 s/round, ~1.5 h per 60-round run.

Changes:
- Fused nuclear move kernels (`evaluate_nuclear`/`apply_nuclear`, used whenever there is no magnetic
  scattering): a move is <= 4 terms (atom phase table x prefactor); per-move phase tables
  E_a[k] = exp(i k theta_a), theta_a = 2 pi (R_a + x_a)/N_a + s_a, replace the Phi gather, the mod-N tables,
  and every sincos (3 lookups per term). The loop runs over observations in chunks, averaging symmetry
  equivalents and accumulating the basis partials in the same pass (no xbuf, no extra parallel regions).
- Displaced-atom structure factors by blocked atom phase tables (`atom_tables`/`accumulate_atoms`):
  E[a, k, i] with atoms contiguous, two complex products per atom-point. Agrees with the old kernel to 1e-13.
- `run(subset=...)` no longer recomputes before each new minibatch (resample recomputes) and takes
  `restore=False`; `anneal` restores the full pool only at checks.

After: 1.2-1.4 ms/move, minibatch recompute 0.5 s, full-pool restore 2.5 s, build 17 -> 5 s;
18 s/round with 24 threads (25 s with Numba's default 32 on this hybrid i9 under WSL; set NUMBA_NUM_THREADS).
Test suite 75 -> 27 s. `test_fused_nuclear_path_matches_general`: identical chains (moves, kinds, chi^2 to 1e-9).
Magnetic moves: `evaluate_magnetic`/`apply_magnetic` (terms carry a magnetic coefficient pair along a spin
vector; spin, swap, displace, pair, exchange). Identical chains to the general kernels
(`test_fused_magnetic_path_matches_general`, nuclear on/off, paramagnetic offset on/off). For bixbyite (32000
undisplaced spins, 100k points) no gain (309 vs 269 us/move, both near the ~100-150 us launch floor), so
`use_fused` defaults to: no magnetic scattering, or any displacive site.

End-to-end `examples/co3.py` (seed 1, NUMBA_NUM_THREADS=24): 18 min wall including data loading (run 13: ~1.5 h);
best round 54, score 0.959, held-out on 31.73 / off 17.34 (run 13: 0.960, 31.28 / 17.56; seeds 31.3-31.7 / 17.4-17.7).
Remaining: bixbyite build 78 s and full-pool resample 27 s on 7M points (gathers), launch overhead for small batches.

## Phase 4 GUI, first slice (2026-11-15)

`src/rmc_discord/gui` (qtpy on PySide6 + pyvistaqt + matplotlib), layout/look of NeuXtalViz (qt views):
`BaseView` = NeuXtalVizWidget (camera buttons, Direction/Manual View tabs, Reciprocal Lattice/Show Axes/Parallel
Projection, plotter, Lattice Parameters/Supercell info tabs, status + progress bar), tools in tabs on the right,
model/view/presenter with Worker/ThreadPool, light theme as a style sheet (pyqtdarktheme unsupported on Py 3.13;
used if installed). App switch: workflow sidebar (1 Structure, 2 Data, 3 Refinement, 4 Correlations,
5 Recalculation), steps enabled when their inputs exist, check mark when done; Applications menu kept.
Shared `Session` (QObject signals) carries crystal, supercell, sample, refinement, correlations.

Done: Structure (load CIF, editable lattice + occupancies, check disordered sites, rigid groups / ions / moments,
N + seed, build in worker, kinds table, unit cell (active opaque) or supercell 3D), Data (data/norm/subtract NXS,
commensurate sampling with Bragg/ring/|Q| masks + error floor, 3D Q cloud, slices on the projection axes with true
metric, Mantid-style axis names). CO3 pipeline reproduces the scripted recipe (3840 atoms, 2,647,090 points).
Entry point `rmc-discord-gui` (extra `gui`); tests/test_gui.py offscreen (60 tests).
All five applications done (2026-11-15):
- Refinement: Setup (nuclear/magnetic, radiation, offset, move weights, displacive sites, penalty/step/cone,
  radial/constant background, symmetrize ops, held-out fraction or count, plane strata + pool off:on ratio) ->
  Initialize (worker); Anneal (Schedule fields, Run/Stop, save/load configuration npz, live score + held-out/pool
  chi^2 plots, records table); 3D configuration colored by kind label, spins as arrows. `anneal` gained
  `callback(row, result)` and `stop()` (polled each round).
- Correlations: quantity (available from the supercell), symmetrize, normalize, current configuration + added npz
  states (error bars), pair/r filters, shells plot + table, 3D pair-vector map (size/color by value), HDF5/CSV.
- Recalculation: data plane on the projection axes, r_max (default 0.99 max_range), |Q| window, Bragg exclusion,
  scale + polynomial background in |Q|, data vs model side by side (true metric), Pearson r, 3D plane in Q.
CO3 through the GUI reproduces the scripts: pool 618,928 / held-out baselines 36.31, 16.6; 4-run displacement
recalculation at l=2.82 Pearson r 0.683 (0.698 diffuse). 62 tests.
Open: session save/restore, refinement 'done' mark for loaded configurations, NeuXtalViz-like volume slicer for
model vs data in 3D.

### GUI: current NeuXtalViz look (NeuXtalViz-tools)

The current NeuXtalViz is `neutrons/NeuXtalViz-tools` (the older
`neutrons/NeuXtalViz` uses the unmaintained pyqtdarktheme). Ported from it:
QDarkStyle light/dark (system-detected; the "dark" plot theme switches the
whole app, synced across views), qtawesome Font Awesome 6 icons (themed,
redrawn on theme change), Rotate View tab (roll/elevation/azimuth steps,
camera readout), console with Expand Console (task stdout/logging routed per
worker thread, not a global redirect), Stop Process in the status bar,
red/green/blue a, b, c buttons, last-directory memory for file dialogs.
