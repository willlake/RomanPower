"""Per-term (PPP / BB / PT) isolation of the windowed-vs-periodic (3,3) covariance
discrepancy seen in test_fkp3_covariance_periodic_approx. Run from tests/:

    python investigate_bb.py
"""

import numpy as np
from jax import numpy as jnp

from romanpower import (MeshAttrs, BinMesh3SpectrumPoles, Mesh3SpectrumPoles,
                      interpolate_window_function)
from romanpower import types
from romanpower.types import Mesh3SpectrumPole
from romanpower.cov3 import compute_spectrum3_covariance
from test_cov3 import get_theory, dirname


mattrs = MeshAttrs(boxsize=2000., boxcenter=[0., 0., 1200.], meshsize=64)
pattrs = mattrs.clone(boxsize=1000., meshsize=64)
theory_full = get_theory(kmax=mattrs.knyq.max())

window2 = types.read(dirname / 'window_fkp2_cov.h5')
window3 = types.read(dirname / 'window_fkp3_cov.h5')
coords = jnp.logspace(-3, 4, 1024)
window2 = interpolate_window_function(window2, coords=coords, order=3)
window3 = window3.map(lambda pole: pole.unravel())
window3 = interpolate_window_function(window3, coords=coords, order=3)

# Bispectrum-only observable: restricts the assembly to the (3,3) block.
bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01},
                             ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
observable3 = Mesh3SpectrumPoles([
    Mesh3SpectrumPole(k=bin3.xavg, k_edges=bin3.edges, nmodes=bin3.nmodes[ill],
                      num_raw=jnp.zeros_like(bin3.xavg[..., 0]), basis=bin3.basis, ell=ell)
    for ill, ell in enumerate(bin3.ells)
])
observable = types.ObservableTree([observable3], fields=[(0, 0, 0)])
nbins = len(np.asarray(bin3.xavg))  # per-ell size
k = np.asarray(bin3.xavg)[..., 0]


def select(sizes):
    # Keep only the given theory orders: the assembly skips terms whose theory is None.
    def theory(fields):
        return theory_full(fields) if len(fields) in sizes else None
    return theory


def report(name, v_win, v_box):
    print(f'--- {name} ---', flush=True)
    for ill, ell in enumerate(bin3.ells):
        s = slice(ill * nbins, (ill + 1) * nbins)
        dw, db = np.diag(v_win)[s], np.diag(v_box)[s]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dw / np.where(db != 0, db, np.nan)
        print(f'  ell={ell} diag win = {dw}')
        print(f'  ell={ell} diag box = {db}')
        print(f'  ell={ell} ratio win/box = {np.array2string(ratio, precision=3)}', flush=True)


cases = [
    ('BB (B-only, no shotnoise)', select({3}), 0.),
    ('PPP (P-only, no shotnoise)', select({2}), 0.),
    ('PPP+PT (P+T, no shotnoise)', select({2, 4}), 0.),
    ('FULL (P+B+T, shotnoise=1/nbar)', theory_full, 1. / 1e-4),
]

values = {}
for name, th, sn in cases:
    cov_win = compute_spectrum3_covariance(window2, window3, observable, theory=th,
                                           shotnoise=sn, cache={}, batch_size=16)
    cov_box = compute_spectrum3_covariance(pattrs, pattrs, observable, theory=th, shotnoise=sn, cache={})
    values[name] = (np.asarray(cov_win.value()), np.asarray(cov_box.value()))
    report(name, *values[name])

# PT is the P x T cross-term: isolate by subtraction.
pt = tuple(vpt - vp for vpt, vp in zip(values['PPP+PT (P+T, no shotnoise)'],
                                       values['PPP (P-only, no shotnoise)']))
report('PT (= (P+T) - (P-only))', *pt)

print('k =', np.array2string(k, precision=3))
print('DONE')
