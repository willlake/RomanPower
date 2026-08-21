"""Example end-to-end workflow: fit bias parameters on a measured P + B data
vector with a periodic covariance, then produce the windowed covariance.

1. Measure the power spectrum multipoles P_ell(k) and the bispectrum in the
   Sugiyama basis B_{l1 l2 L}(k1, k2) from a mock (here: an anisotropic
   Gaussian mesh with Kaiser multipoles -- replace with your FKP catalog
   measurement, cf. tests/test_mesh2.py / tests/test_mesh3.py for the
   normalization and shot-noise conventions).
2. Fit the tracer bias parameters (here b1, b2) on the joint (P, B) data
   vector, using the PERIODIC-box P + B covariance matrix:
   compute_spectrum3_covariance(mattrs, mattrs, ...), i.e. the analytic
   Sugiyama et al. formulas (PP / PB / PPP+BB+PT terms).
3. Feed the best-fit theory into the estimation of the full covariance with
   non-trivial window2 / window3 (computed from the survey randoms):
   compute_spectrum3_covariance(window2, window3, ...).

Run with the cosmodesi environment:
    python scripts/example_fit_bias_covariance.py
"""

from pathlib import Path

import numpy as np
import jax
from jax import numpy as jnp

from jax import random

from romanpower import (MeshAttrs, BinMesh2SpectrumPoles, BinMesh3SpectrumPoles,
                      compute_mesh2_spectrum, compute_mesh3_spectrum,
                      generate_anisotropic_gaussian_mesh, generate_uniform_particles,
                      FKPField, compute_fkp2_shotnoise, compute_fkp3_shotnoise,
                      interpolate_window_function, read)
from romanpower import types
from romanpower.cov3 import compute_spectrum3_covariance, compute_fkp3_covariance_window
from romanpower.cov2 import compute_fkp2_covariance_window
from romanpower.pt import (prepare_spectrum2_redshift_tracer, spectrum2_redshift_tracer,
                         spectrum3_redshift_tracer, spectrum4_redshift_tracer,
                         ProjectToPoles, ProjectToSell)


dirname = Path(__file__).parent.parent / 'tests' / '_tests'


# ------------------------------------------------------------------
# Setup: geometry, fiducial cosmology, fiducial bias parameters
# ------------------------------------------------------------------
mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200.], meshsize=64)
pattrs = mattrs.clone(boxsize=1000., meshsize=64)   # survey volume (randoms)
nbar = 1e-4                                          # mean tracer density
shotnoise = 1. / nbar
f = 0.8                                              # growth rate (fixed here)

from cosmoprimo.fiducial import DESI
cosmo = DESI(engine='eisenstein_hu')
kt = np.linspace(0.001, 0.3, 201)
pkt = cosmo.get_fourier().pk_interpolator().to_1d(z=0.)(kt)

def pk_callable(q):
    return jnp.interp(q, kt, pkt)

pknow_callable = pk_callable  # no-wiggle spectrum; plug a BAO-filtered one if desired

k_table = jnp.logspace(-3, np.log10(mattrs.knyq.max()), 80)
table, table_now = prepare_spectrum2_redshift_tracer(k_table, pk_callable, pknow_callable)

# Fiducial bias parameters; (b1, b2) are fitted below, the rest stay fixed.
fid_bias = {"b1": 2.0, "b2": 0.5, "bs": -0.3, "b3nl": 0.1,
            "c1": 0.1, "c2": 0.2, "X_FoG": 2., "snb0": 0.1, "sn0": 0.1}


def get_bias(params):
    return {**fid_bias, "b1": params[0], "b2": params[1]}


def make_theory(bias):
    """3D P / B / T callables consumed by compute_spectrum3_covariance."""
    bias_params = {0: bias}

    def P(kvec):
        return spectrum2_redshift_tracer(kvec, table, table_now, f, bias_params)

    def B(k1vec, k2vec, k3vec):
        return spectrum3_redshift_tracer(k1vec, k2vec, pk_callable, pknow_callable,
                                         f=f, bias_params=bias_params)

    def T(k1vec, k2vec, k3vec, k4vec):
        return spectrum4_redshift_tracer(k1vec, k2vec, k3vec, pk_callable, pknow_callable,
                                         f=f, bias_params=bias_params)

    def theory(fields):
        return {2: P, 3: B, 4: T}.get(len(fields), None)

    return theory


# ------------------------------------------------------------------
# 1) "Measurement": P_ell(k) and Sugiyama B_{l1 l2 L}(k1, k2) from a mock
# ------------------------------------------------------------------
# Kaiser multipoles for the mock's input power spectrum. Cast to the mesh
# dtype: float64 pole values would silently promote the generated mesh to
# float64 and break compute_mesh3_spectrum's complex64 scan carry.
b1 = fid_bias["b1"]

def kaiser_pole(coef):
    return lambda k: (coef * pk_callable(k)).astype(mattrs.rdtype)

poles_in = {0: kaiser_pole(b1**2 + 2. / 3. * b1 * f + f**2 / 5.),
            2: kaiser_pole(4. / 3. * b1 * f + 4. / 7. * f**2),
            4: kaiser_pole(8. / 35. * f**2)}
# los must be 'local' or 'z': the theory kernels and the covariance
# projection (S_{l1 l2 L} with the LOS along z) assume this convention, and
# the mock's RSD axis must match the measurement line of sight.
los = 'z'
mesh = generate_anisotropic_gaussian_mesh(mattrs, poles_in, seed=42, los=los, unitary_amplitude=True)

# Sample an actual FKP catalog from the mesh (uniform particles weighted by
# 1 + delta, plus randoms): the measurement then carries the same shot
# noise 1/nbar that is passed to the analytic covariance below -- a plain
# mesh measurement would NOT (no particles, no shot noise).
size = int(nbar * pattrs.boxsize.prod())
seeds = random.split(random.key(68))
data_p = generate_uniform_particles(pattrs, size, seed=seeds[0]).clone(attrs=mattrs)
data_p = data_p.clone(weights=1. + mesh.read(data_p.positions, resampler='cic', compensate=True))
randoms_p = generate_uniform_particles(pattrs, 10 * size, seed=seeds[1]).clone(attrs=mattrs)
fkp = FKPField(data_p, randoms_p)
kw = dict(resampler='tsc', interlacing=3, compensate=True)
fmesh = fkp.paint(**kw, out='complex')
# Residual shot power in the FKP field, passed to the covariance:
# (1 + alpha) / nbar with alpha the data-to-randoms weight ratio.
alpha = float(data_p.weights.sum() / randoms_p.weights.sum())
shotnoise = (1. + alpha) / nbar

bin2 = BinMesh2SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01}, ells=(0, 2, 4))
norm2 = nbar**2 * pattrs.boxsize.prod()
spectrum2 = compute_mesh2_spectrum(fmesh, bin=bin2, los=los).clone(
    norm=[norm2] * len(bin2.ells), num_shotnoise=compute_fkp2_shotnoise(fkp, bin=bin2))

bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01},
                             ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
norm3 = nbar**3 * pattrs.boxsize.prod()
num_shotnoise3 = compute_fkp3_shotnoise(fkp, bin=bin3, los=los, **kw)
spectrum3 = compute_mesh3_spectrum(fmesh, bin=bin3, los=los)
spectrum3 = spectrum3.map(lambda pole: pole.clone(norm=norm3)).clone(num_shotnoise=num_shotnoise3)

# The measured observables define both the data vector and the covariance
# binning. NOTE: the underlying field is Gaussian, so the (shot-noise
# subtracted) bispectrum has essentially no signal -- with real data
# replace the mesh/catalog construction above by your survey catalogs.
observable = types.ObservableTree([spectrum2, spectrum3], fields=[(0, 0), (0, 0, 0)])
data = np.concatenate([np.asarray(obs.value()).ravel() for _, obs in observable.items(level=None)])


# ------------------------------------------------------------------
# 2) Fit (b1, b2) with the periodic P + B covariance
# ------------------------------------------------------------------
# Periodic covariance: pass MeshAttrs in place of (window2, window3); the
# theory is evaluated at the fiducial bias (optionally iterate the fit).
print('Computing the periodic P + B covariance...')
cov_box = compute_spectrum3_covariance(pattrs, pattrs, observable,
                                       theory=make_theory(get_bias([fid_bias["b1"], fid_bias["b2"]])),
                                       shotnoise=shotnoise, cache={})
C = np.asarray(cov_box.value())
Cinv = jnp.asarray(np.linalg.inv(C))

# Binned model multipoles, evaluated at the measured k coordinates.
to_poles = ProjectToPoles(ells=(0, 2, 4), mu=10)
k2d = np.asarray(bin2.xavg)                                # (nk,)
mu = np.asarray(to_poles.mu)
kvec_P = k2d[:, None, None] * np.stack([np.sqrt(1. - mu**2), np.zeros_like(mu), mu], axis=-1)  # (nk, nmu, 3)

to_Sell = ProjectToSell(ells=bin3.ells, size=6)
k3d = np.asarray(bin3.xavg)                                # (nbins, 2) paired (k1, k2)
k1vec_B = k3d[:, 0, None, None] * np.asarray(to_Sell.k1hat)[None, ...]   # (nbins, nnodes, 3)
k2vec_B = k3d[:, 1, None, None] * np.asarray(to_Sell.k2hat)[None, ...]


def model_vector(params):
    bias_params = {0: get_bias(params)}
    P3d = spectrum2_redshift_tracer(jnp.asarray(kvec_P), table, table_now, f, bias_params)   # (nk, nmu)
    p_poles = to_poles(P3d)                                                # (3, nk)
    B3d = spectrum3_redshift_tracer(jnp.asarray(k1vec_B), jnp.asarray(k2vec_B),
                                    pk_callable, pknow_callable, f=f, bias_params=bias_params)
    b_poles = to_Sell(B3d)                                                 # (2, nbins)
    return jnp.concatenate([p_poles.ravel(), b_poles.ravel()])


@jax.jit
def chi2(params):
    r = jnp.asarray(data) - model_vector(params)
    return r @ Cinv @ r


print('Fitting (b1, b2)...')
from scipy import optimize
res = optimize.minimize(jax.value_and_grad(chi2), x0=np.array([1.8, 0.]),
                        jac=True, method='L-BFGS-B')
# NOTE: with the Gaussian mock above the bispectrum data is just a (large)
# noise realization inconsistent with the tree-level B(b1, b2) model, so the
# joint fit is pulled and the chi2 is poor -- with a real (non-Gaussian)
# catalog this recovers the tracer's bias parameters.
print(f'  best fit: b1 = {res.x[0]:.3f}, b2 = {res.x[1]:.3f}, chi2 = {res.fun:.1f} '
      f'({data.size} data points)')


# ------------------------------------------------------------------
# 3) Windowed covariance at the best-fit theory
# ------------------------------------------------------------------
# Covariance windows from the survey randoms (here: uniform randoms filling
# pattrs, as a stand-in for your survey selection). Cached on disk.
randoms = generate_uniform_particles(pattrs, int(nbar * pattrs.boxsize.prod()), seed=32).clone(attrs=mattrs)
sedges = {'step': 40.}

window2_fn = dirname / 'window_fkp2_cov.h5'
if window2_fn.exists():
    window2 = read(window2_fn)
else:
    window2 = compute_fkp2_covariance_window(randoms, edges=sedges, interlacing=2, resampler='tsc',
                                             los='local', group_sizes=(2, 3, 4), max_total_size=6, ells=[0, 2, 4])
    window2.write(window2_fn)

window3_fn = dirname / 'window_fkp3_cov.h5'
if window3_fn.exists():
    window3 = read(window3_fn)
else:
    window3 = compute_fkp3_covariance_window(randoms, edges=sedges, interlacing=2, resampler='tsc',
                                             los='local', buffer_size=50, ells=[(0, 0, 0)])
    window3.write(window3_fn)

coords = jnp.logspace(-3, 4, 1024)
window2 = interpolate_window_function(window2, coords=coords, order=3)
window3 = window3.map(lambda pole: pole.unravel())
window3 = interpolate_window_function(window3, coords=coords, order=3)

print('Computing the windowed P + B covariance at the best-fit theory...')
cov_win = compute_spectrum3_covariance(window2, window3, observable,
                                       theory=make_theory(get_bias(res.x)),
                                       shotnoise=shotnoise, cache={}, batch_size=16)
cov_win.write(dirname / 'covariance_pb_windowed.h5')
print(f"Windowed covariance written to {dirname / 'covariance_pb_windowed.h5'}")

# Quick sanity plot of the two diagonals (periodic vs windowed).
if False:
    kw = dict(yscale='log')
    fig = cov_win.plot_diag(**kw, color='C0', show=False)
    cov_box.plot_diag(**kw, fig=fig, color='C1', show=True)
