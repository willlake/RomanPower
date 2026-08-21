"""BB-term windowed-vs-periodic ratio as a function of the angular quadrature
order (COV3_QUAD_SIZE env var, read by cov3 at call time). Run from tests/:

    COV3_QUAD_SIZE=8 python investigate_bb_q.py
"""

import os

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

bin3 = BinMesh3SpectrumPoles(mattrs, edges={'step': 0.01, 'min': 0.01},
                             ells=[(0, 0, 0), (2, 0, 2)], basis='sugiyama-diagonal')
observable3 = Mesh3SpectrumPoles([
    Mesh3SpectrumPole(k=bin3.xavg, k_edges=bin3.edges, nmodes=bin3.nmodes[ill],
                      num_raw=jnp.zeros_like(bin3.xavg[..., 0]), basis=bin3.basis, ell=ell)
    for ill, ell in enumerate(bin3.ells)
])
observable = types.ObservableTree([observable3], fields=[(0, 0, 0)])
nbins = len(np.asarray(bin3.xavg))


def theory_B(fields):
    return theory_full(fields) if len(fields) == 3 else None


qsize = os.environ.get('COV3_QUAD_SIZE', '6')
cov_win = compute_spectrum3_covariance(window2, window3, observable, theory=theory_B,
                                       shotnoise=0., cache={}, batch_size=16)
cov_box = compute_spectrum3_covariance(pattrs, pattrs, observable, theory=theory_B, shotnoise=0., cache={})
v_win, v_box = np.asarray(cov_win.value()), np.asarray(cov_box.value())

print(f'--- BB-only, COV3_QUAD_SIZE={qsize} ---')
for ill, ell in enumerate(bin3.ells):
    s = slice(ill * nbins, (ill + 1) * nbins)
    dw, db = np.diag(v_win)[s], np.diag(v_box)[s]
    print(f'  ell={ell} diag win = {dw}')
    print(f'  ell={ell} diag box = {db}')
    print(f'  ell={ell} ratio win/box = {np.array2string(dw / db, precision=3)}', flush=True)
print('DONE')
