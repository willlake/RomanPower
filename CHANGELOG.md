# Changelog

All notable changes to `RomanPower` are recorded here. The package is a rename-fork of
[`jax-power`](https://github.com/adematti/jax-power) by Arnaud de Mattia; entries below
cover changes made in this fork, not upstream ones ported over.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `BinMesh3CorrelationPoles(buffer_size=3)` — the DESI fiducial for LRG — asks for ~79 GB on
  a 750³ mesh in double precision, which does not fit even an 80 GB GPU in a single process.
  `buffer_size` trades memory for speed without changing the result; drop it to 0 there, or
  distribute the mesh over several devices.

## [0.2.0] — 2026-08-21

Initial fork of `jax-power`, renaming the package `jaxpower` → `romanpower`
(`6435e35`, `c11b14e`), and a fix to `nb/mesh2.ipynb` (`ce4f6a1`). The API, module layout
and conventions follow upstream so that upstream fixes can be ported over directly.

[Unreleased]: https://github.com/willlake/RomanPower/compare/v0.2.0...HEAD
