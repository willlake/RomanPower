# Changelog

All notable changes to `RomanPower` are recorded here. The package is a rename-fork of
[`jax-power`](https://github.com/adematti/jax-power) by Arnaud de Mattia; entries below
cover changes made in this fork, not upstream ones ported over.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`romanpower/thecov.py`: analytic covariance of $P_{0,2,4}(k)$** (Gaussian term with the survey
  window, regular trispectrum $T_0$ with its shot-noise part, super-sample covariance), a JAX port of
  [`thecov`](https://github.com/cosmodesi/thecov) / CovaPT (Wadekar & Scoccimarro 2020) and of the
  $T_0$ of Kobayashi (2023), built on `MeshAttrs` / `ParticleField.paint` / `_bincount`. Opt-in
  (`from romanpower import thecov`; it switches `jax_enable_x64` on). Entry points:
  `SurveyWindow` (windows $W_{22}, W_{12}, W_{10}$ and their Cartesian moments from a random
  catalogue), `compute_window_kernels` + `gaussian_covariance_survey`, `gaussian_covariance_box`,
  `compute_trispectrum_covariance`, `compute_ssc_covariance`, `pbs_bias_parameters`. Tests in
  `tests/test_thecov.py`; the validation against `thecov`, CovaPT and `powercovfft` on the DESI
  LRG NGC $0.4<z<0.6$ geometry (and against 985 mocks) is documented in
  `~/Projects/Roman/Thecov/thecov_jax/VALIDATION.md`. Differences with `thecov` are deliberate and
  documented there: CovaPT's cross-multipole blocks are stored untransposed (thecov transposes
  them; the mocks confirm CovaPT), the `W12_k1L4` moment is complete (CovaPT and thecov drop
  three cubic terms), the $T_0$ cross-multipole blocks keep $L_{\ell_1}$ on $\hat k_1$
  (`powercovfft` swaps the multipoles together with the momenta in its $k_1 \leftrightarrow k_2$
  terms), the SSC window multipoles are built on the correct $\hat k$ grid (thecov's `compute_window_power`
  transposes it), and the $T_0$ angular integrals are done by exact / log-spaced Gauss quadrature rather
  than FFTLog. By default (`exact_high_order=True`) the 6th and 8th order window moments are painted
  and the $L_2 L_4$ / $L_4 L_4$ window products are exact, where CovaPT and thecov use a $\hat k_2 \to
  \hat k_1$ approximation that makes the three-multipole Gaussian matrix indefinite
  ($\mathrm{tr}(C_\mathrm{ana}^{-1} C_\mathrm{mock})/N \simeq 4.5$ instead of 1.1 on the DESI LRG mocks; see
  VALIDATION.md section 6); `exact_high_order=False` reproduces their kernels.
  `MeshWindow.from_randoms` builds $W_{22} = (\alpha n_r w_c w_\mathrm{FKP})^2$ as the product of two
  random halves painted on a fine mesh (the window of the FKP field, i.e. of the *weighted* density), where
  thecov's catalogue sum $\alpha \sum_r n(z_r) w_c w_\mathrm{FKP}^2$ is one power of the completeness weight
  short when `NZ` is the number density: on the DESI LRG randoms this raises the Gaussian variance by 9%
  and closes most of the "analytic variance 10% low" gap to the mocks: that deficit was the window
  definition, not the kernels, and `cov2.py` -- which squares the painted weighted density and whose
  Gaussian variance was 9% higher -- was right. A Gaussian-mock closure test with the real window
  reproduces the mock variance to 0.2-0.5% (VALIDATION.md section 7).

### Fixed

- **`nmodes` overflowed `int32` in the bispectrum binning**, silently corrupting
  `compute_mesh3_spectrum` beyond a mesh-dependent scale (`romanpower/mesh3.py`).

  `_make_edges3` built the mode counts of `BinMesh3SpectrumPoles` and
  `BinMesh3CorrelationPoles` as a product of the per-axis counts,
  `jnp.prod(_product(nmodes1d)[mask], axis=-1)`. The per-axis counts are small, but their
  product is not, and without `jax_enable_x64` they are `int32`: on a 750³ mesh the
  product passes `2**31 - 1` at only the 7th $k$ bin, $k \simeq 0.033$, where the true
  count is 2 297 668 356. It wrapped there to a negative number, with no warning.

  `compute_mesh3_spectrum` divides the raw bispectrum by `bin.nmodes`, so **every bin from
  that point on came out wrong** — wrong by the ratio of the wrapped to the true count, and
  frequently of the opposite sign. On the DESI DR2 `holi-v4-altmtl` LRG NGC z0.4–0.6 mock
  the measured $B_{000}$ and $B_{202}$ disagreed with the reference by a median factor of
  ~500 above $k = 0.033$, while agreeing to five digits below it.

  The counts are now promoted before the product: to `int64` when `jax_enable_x64` is set
  (exact, and what the DESI production runs already got), and otherwise to `float32`, the
  widest type JAX offers in that mode, which still resolves them to ~5e-8 relative — far
  below the estimator's own single-precision noise. With the fix, the same mock reproduces
  the reference to a median relative difference of 1.6e-5 in single precision, which is the
  float32 noise floor.

  The 2-point path was never affected: `BinMesh2SpectrumPoles` counts modes with a single
  `bincount` rather than a product, so it is bounded by the number of mesh cells. The
  3-point *correlation* values were not affected either, since
  `BinMesh3CorrelationPoles.__call__` performs a Bessel transform and never divides by
  `nmodes`; only the `nmodes` it recorded was wrong, and that is now fixed too.

- **Pathological GPU slowdown in mesh $k$-binning** (`romanpower/mesh.py`, commit
  `a084351`, branch `gpu-bincount-contention`).

  XLA:GPU lowers a scatter-add into atomics on the destination buffer. Binning a mesh sends
  every cell (16.7e6 for 256³) into a couple of hundred $k$-bins, and the resulting
  contention ran three orders of magnitude off memory bandwidth — enough that an A100 was
  slower than a 3-core login node. This dominated every spectrum measurement on GPU.

  Each bin now gets `nsub` private accumulators, summed at the end; the enlarged table
  still fits in L2. Applied in `_bincount` and, for bin construction, `_get_bin_attrs`.
  Measured on an A100 (256³ mesh, 5.9e6 galaxies):

  | | before | after |
  |---|---|---|
  | `BinMesh2SpectrumPoles` constructor | 372.58 s | 2.10 s |
  | `bincount` microbenchmark | 2.2 s | 0.0009 s |
  | $P_\ell(k)$, $\ell = 0, 2, 4$ | ~70 s | 0.16 s cached |

  Painting is unaffected — it scatters into 16.7e6 cells rather than a few hundred, so it
  never enters the contention regime. The change is gated on
  `jax.default_backend() == 'gpu'`, leaving CPU execution bit-for-bit unchanged, and
  setting `romanpower.mesh._bincount_nsub = 1` restores the previous behaviour at runtime.

### Known issues

- **`pt.spectrum4_redshift_tracer` is not a correct tree-level trispectrum** (found while
  validating `thecov.py`, which does not use it; `romanpower/pt.py` is left unchanged here):
  (i) `_F3_unsym` / `_G3_unsym` are not the standard recursion of Bernardeau et al. (2002,
  Eqs. 43-44) -- they have a single term $7\alpha(k_{12}, k_3) F_2(k_1, k_2) + 2\beta(k_{12}, k_3)
  G_2(k_1, k_2)$ instead of $7\alpha(k_1, k_{23}) F_2(k_2, k_3) + 2\beta(k_1, k_{23}) G_2(k_2, k_3)
  + G_2(k_1, k_2)[7\alpha(k_{12}, k_3) + 2\beta(k_{12}, k_3)]$ -- so the symmetrized $F_3(k, q, -q)$
  does not reproduce the $P_{13}$ kernel (it has the opposite sign for $q \sim k$); the version in
  `thecov._PTKernels` does, to 7 digits. (ii) The parallelogram branch (`_para_channel`, used
  for $T(k, -k, k', -k')$) sums 8/8/16-weighted snake terms over *both* channels
  $q = k \pm k'$, which is exactly twice the explicit 12-permutation sum of Eq. (4) of Kobayashi
  (2023) (the 8/8/16 weights of his Eq. (6) already stand for both channels through the angular
  symmetry). (iii) `_Z3` lacks the $b_2 F_2(k_2, k_3)$ and $2 b_{G_2} K_{G_2}(k_1, k_{23}) F_2(k_2,
  k_3)$ third-order bias terms. Diagnostics: `~/Projects/Roman/Thecov/thecov_jax/debug_t0.py`.

- `tests/test_mesh3.py::test_fkp3_shotnoise` fails under `jax_enable_x64 = False`. The test
  compares against hard-coded reference values with `np.allclose` at its default
  tolerances, which single precision does not meet; the test file's `__main__` block sets
  x64 on. Pre-existing, unrelated to the fixes above.
- The window multipoles $Q_\ell$ are sums of spherical Bessel functions over the Fourier
  mesh, and the cancellation grows sharply with $\ell$. Measured against the DESI DR2
  reference on a 750³ mesh, `max|diff|/max|ref|` over $1 < s < 3000$ Mpc/$h$:

  | | $Q_0$ | $Q_2$ | $Q_4$ | $Q_6$ | $Q_8$ | window matrix |
  |---|---|---|---|---|---|---|
  | float64 | 1.1e-14 | 1.7e-14 | 1.3e-10 | 6.3e-07 | 1.7e-02 | 3.6e-06 |
  | float32 | 2.3e-05 | 2.4e-04 | 9.6e-02 | 1.0e+03 | 8.8e+06 | 4.1e+01 |

  Single precision is unusable for $Q_{\ell \geq 4}$ and for any matrix built from them; run
  window computations with `jax_enable_x64 = True`, as the DESI production scripts do. Even
  in double precision $Q_8$ degrades at small $s$, so treat the highest multipoles with
  care.
- Not a defect, but a sharp edge worth knowing: `ParticleField(exchange=True)`
  redistributes whatever each process brings, so if every process is handed the *whole*
  catalog rather than a slice of it, the global field ends up with `jax.process_count()`
  copies of every particle. Nothing warns. The DESI pipeline never hits this because it
  reads catalogs through MPI, which slices them.

  It shows up asymmetrically, which is what makes it dangerous. Quantities normalized by
  as many field factors as they contain are immune -- power spectra, bispectra, the window
  matrices, and the `WW` term of the covariance window all come out unchanged -- while the
  covariance's `WS`/`SW` keep one net factor of nproc and `SS` two. On the DESI DR2 LRG NGC
  z0.4-0.6 covariance that made sigma ~27% too small, with `WW` looking perfect throughout.
  Measured at a 288^3 mesh:

      nproc=1            WW(0)=5.57880773e-10  WS(0)=1.64662459e-06  SS(0)=4.97600644e-03
      nproc=4 unsliced   WW(0)=5.57880773e-10  WS(0)=4.11656148e-07  SS(0)=3.11000403e-04
      nproc=4 sliced     WW(0)=5.57562003e-10  WS(0)=1.64661812e-06  SS(0)=4.97880745e-03

  With the catalogs sliced, the covariance reproduces the DESI reference: sigma to a median
  of 9.6e-05 and the correlation matrix to 7.9e-04, the residual being the random split
  drawing differently at a different process count.
- `create_sharding_mesh()` with several local devices in a single process hangs in
  `ParticleField.paint` (reproduced at a 64^3 mesh, where painting should take
  milliseconds). One process per GPU via `jax.distributed.initialize()` -- what the DESI
  production scripts use -- works and paints the same case in ~1 s.
- `BinMesh3CorrelationPoles(buffer_size=3)` — the DESI fiducial for LRG — asks for ~79 GB on
  a 750³ mesh in double precision, which does not fit even an 80 GB GPU in a single process.
  `buffer_size` trades memory for speed without changing the result; drop it to 0 there, or
  distribute the mesh over several devices.

## [0.2.0] — 2026-08-21

Initial fork of `jax-power`, renaming the package `jaxpower` → `romanpower`
(`6435e35`, `c11b14e`), and a fix to `nb/mesh2.ipynb` (`ce4f6a1`). The API, module layout
and conventions follow upstream so that upstream fixes can be ported over directly.

[Unreleased]: https://github.com/willlake/RomanPower/compare/v0.2.0...HEAD
