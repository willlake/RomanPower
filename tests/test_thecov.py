"""
Tests of romanpower.thecov (analytic covariance of P_ell: window kernels, T0, SSC).

    python tests/test_thecov.py            # fast tests (CPU is fine)
    python tests/test_thecov.py --slow     # + cube-of-randoms vs periodic box (a few minutes on CPU)

The comparisons with thecov / powercovfft on the DESI LRG geometry live in
~/Projects/Roman/Thecov/thecov_jax (see VALIDATION.md there).
"""
import sys
import time

import numpy as np
import jax
from jax import numpy as jnp

from romanpower import thecov as tj


def get_pk():
    """A smooth, Kaiser-like P_0, P_2, P_4 model (no cosmoprimo needed)."""
    def pk_lin(k):
        k = np.asarray(k, dtype=float)
        return 2e4 * (k / 0.05) / (1 + (k / 0.05)**2.5)  # (Mpc/h)^3, peak ~ 0.05
    f, b1 = 0.8, 1.9
    beta = f / b1

    def poles(k):
        p = b1**2 * pk_lin(k)
        return np.array([(1 + 2 / 3 * beta + 1 / 5 * beta**2) * p, (4 / 3 * beta + 4 / 7 * beta**2) * p, 8 / 35 * beta**2 * p])
    return pk_lin, poles, f, b1


def test_kbins():
    assert tj.count_kbins(0.02, 0.2, 0.005) == 36
    assert tj.count_kbins(0.02, 0.199, 0.001) == 179  # (0.199 - 0.02) / 0.001 = 179.00000000000003
    edges = tj.linear_kedges(0.02, 0.2, 0.005)
    assert len(edges) == 37 and np.isclose(edges[-1], 0.2)


def test_count_modes():
    L = 1000.
    kedges = tj.linear_kedges(0., 0.1, 0.01)
    exact = tj.count_modes(kedges, L)
    approx = tj.count_modes(kedges, L, exact_below=0.)
    assert np.all(exact >= 1)  # the first shell contains at least the k = 0 mode
    assert np.allclose(approx, L**3 / (6 * np.pi**2) * (kedges[1:]**3 - kedges[:-1]**3))
    assert np.isclose(exact.sum(), approx.sum(), rtol=0.03)  # lattice counts fluctuate bin by bin, the total does not
    kfun = 2 * np.pi / L
    # brute force
    n = int(np.ceil(0.1 / kfun))
    ix = np.arange(-n, n + 1)
    ir = np.sqrt(ix[:, None, None]**2 + ix[None, :, None]**2 + ix[None, None, :]**2).ravel() * kfun
    ref, _ = np.histogram(ir, bins=kedges)
    assert np.array_equal(exact, ref)


def test_sample_kmodes():
    L = 2000.
    kedges = tj.linear_kedges(0.02, 0.1, 0.01)
    modes, nmodes = tj.sample_kmodes(kedges, L, max_modes=50, k_shell_approx=0.06, seed=1)
    assert len(modes) == 8 and len(nmodes) == 8
    kfun = 2 * np.pi / L
    for i, m in enumerate(modes):
        assert m.shape[1] == 4 and len(m) <= 50
        assert np.allclose(m[:, 3], np.sqrt((m[:, :3]**2).sum(axis=1)))
        assert np.all(m[:, 3] * kfun >= kedges[i] - 1e-9) and np.all(m[:, 3] * kfun <= kedges[i + 1] + 1e-9)
    assert np.allclose(nmodes[:4], tj.count_modes(kedges, L)[:4])


def test_sample_kmodes_shell_uniform():
    """The continuous-shell branch draws every lattice mode of the shell with the same probability, as the enumerated one."""
    L, kedges = 1000., [0.10, 0.11]
    kfun = 2 * np.pi / L
    n = int(np.ceil(kedges[1] / kfun))
    ix = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(ix, ix, ix, indexing='ij'), axis=-1).reshape(-1, 3)
    kr = np.sqrt((grid**2).sum(axis=1)) * kfun
    shell = grid[(kr >= kedges[0]) & (kr < kedges[1])]
    index = {tuple(p): i for i, p in enumerate(shell)}
    expected = 100
    modes = tj.sample_kmodes(kedges, L, max_modes=expected * len(shell), k_shell_approx=0., seed=2)[0][0]
    counts = np.bincount([index[tuple(p)] for p in modes[:, :3].astype(int)], minlength=len(shell))  # KeyError if outside
    chi2 = np.sum((counts - expected)**2 / expected)
    assert abs(chi2 - len(shell)) < 5 * np.sqrt(2 * len(shell)), (chi2, len(shell))  # uniform radius in the bare shell: 4.8 len(shell)


def test_gaussian_covariance_symmetric():
    """The rows of the kernels come from the sampled k1 modes of their own bin: the Gaussian covariance is symmetrised."""
    from romanpower.mesh import MeshAttrs
    n, L = 32, 2 * np.pi / 0.01
    attrs = MeshAttrs(meshsize=n, boxsize=L, boxcenter=np.array([1.6, 0.4, 0.25]) * L, dtype=float, fft_backend='jax')
    x = (np.indices((n,) * 3) - n / 2 + 0.5) / (n / 2)
    W22 = np.exp(-(x[0]**2 / 0.5 + x[1]**2 / 0.3 + x[2]**2 / 0.2))
    window = tj.MeshWindow(W22, attrs, 12, W12=np.sqrt(W22))
    kedges = tj.linear_kedges(0.02, 0.08, 0.02)
    kernels = tj.compute_window_kernels(window, kedges, kmodes_sampled=32, batch_size=8, progress=False, seed=1, exact_high_order=False)
    _, poles, _, _ = get_pk()
    cov = tj.blocks_to_matrix(tj.gaussian_covariance_survey(kernels, poles(kedges[:-1] + 0.01), alpha=0.1, I22=window.I('22')))
    assert np.array_equal(cov, cov.T)


def test_gaussian_box_vs_cov2():
    """Box Gaussian formula against romanpower.cov2.compute_spectrum2_covariance(MeshAttrs, flags=('smooth',))."""
    from romanpower import MeshAttrs, compute_spectrum2_covariance
    from romanpower.types import Mesh2SpectrumPole, Mesh2SpectrumPoles
    pk_lin, poles, f, b1 = get_pk()
    mattrs = MeshAttrs(boxsize=1000., meshsize=64)
    kedges = tj.linear_kedges(0.02, 0.15, 0.01)
    k = kedges[:-1] + 0.005
    pk = poles(k)
    shotnoise = 1e3
    edges = np.column_stack([kedges[:-1], kedges[1:]])
    # value() of the poles is num_raw - num_shotnoise: give cov2 the shot-noise-including P0 as num_raw
    spectrum = Mesh2SpectrumPoles([Mesh2SpectrumPole(k=k, k_edges=edges, num_raw=pk[i] + shotnoise * (ell == 0), num_shotnoise=np.zeros_like(k), ell=ell)
                                   for i, ell in enumerate((0, 2, 4))])
    ref = compute_spectrum2_covariance(mattrs, spectrum, flags=('smooth',)).value()
    nmodes = mattrs.boxsize.prod() / (6 * np.pi**2) * (kedges[1:]**3 - kedges[:-1]**3)
    mine = tj.blocks_to_matrix(tj.gaussian_covariance_box(pk, nmodes, shotnoise=shotnoise))
    assert np.allclose(mine, ref, rtol=1e-8, atol=0.), np.abs(mine / ref - 1).max()


def test_contractions():
    """Tensor contractions used for the window multipoles against brute force."""
    rng = np.random.default_rng(0)
    W2 = rng.normal(size=(6, 4)); W4 = rng.normal(size=(15, 4))
    a, b, c, d = [rng.normal(size=3) for _ in range(4)]
    T2 = np.zeros((3, 3, 4)); T4 = np.zeros((3, 3, 3, 3, 4))
    for m, (i, j) in enumerate(tj._IDX2):
        T2[i, j] = T2[j, i] = W2[m]
    import itertools
    for m, idx in enumerate(tj._IDX4):
        for perm in set(itertools.permutations(idx)):
            T4[perm] = W4[m]
    assert np.allclose(np.asarray(tj._contract2(jnp.array(W2), a, b)), np.einsum('ijk,i,j->k', T2, a, b))
    assert np.allclose(np.asarray(tj._contract4(jnp.array(W4), a, b, c, d)), np.einsum('ijklm,i,j,k,l->m', T4, a, b, c, d))


def test_pt_kernels():
    """Z2 against RomanPower's pt._Z2 (different bias basis) and Z3 symmetry / squeezed limits."""
    from romanpower import pt
    f = 0.7
    bias = tj.pbs_bias_parameters(2.0)
    kern = tj._PTKernels(f, bias)
    rng = np.random.default_rng(2)
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    k1, k2 = rng.normal(size=3) * 0.1, rng.normal(size=3) * 0.1
    # bs (mu^2 - 1/3) basis: bs = 2 bG2, b2' = b2 - 4/3 bG2
    pt_bias = {'b1': bias['b1'], 'b2': bias['b2'] - 4. / 3. * bias['bG2'], 'bs': 2. * bias['bG2'], 'c1': 0., 'c2': 0., 'snb0': 0., 'sn0': 0., 'X_FoG': 0.}
    # pt uses the z axis as line of sight: rotate
    e3 = n; e1 = np.cross(e3, [0.3, 0.5, 0.8]); e1 /= np.linalg.norm(e1); e2 = np.cross(e3, e1)
    R = np.array([e1, e2, e3])
    k1r, k2r = R @ k1, R @ k2
    # evaluate pt's Z2 through the bispectrum? simpler: use its internal via spectrum3? Not exposed. Test Z2 through
    # the parallelogram trispectrum's snake structure instead: compare Z2 directly with a re-implementation of pt._Z2
    n1, n2 = np.linalg.norm(k1), np.linalg.norm(k2)
    mu1, mu2 = k1 @ n / n1, k2 @ n / n2
    x12 = k1 @ k2 / (n1 * n2)
    km = n1 * mu1 + n2 * mu2
    F2 = 5. / 7. + x12 / 2. * (n1 / n2 + n2 / n1) + 2. / 7. * x12**2
    G2 = 3. / 7. + x12 / 2. * (n1 / n2 + n2 / n1) + 4. / 7. * x12**2
    b1, b2p, bs = pt_bias['b1'], pt_bias['b2'], pt_bias['bs']
    Z2_pt = (b2p / 2. + bs / 2. * (x12**2 - 1. / 3.) + km / 2. * (mu1 / n1 * f * (b1 + f * mu2**2) + mu2 / n2 * f * (b1 + f * mu1**2))
             + b1 * F2 + f * km**2 / (n1**2 + n2**2 + 2 * n1 * n2 * x12) * G2)
    Z2_mine = float(kern.Z2(jnp.array(k1), jnp.array(k2), jnp.array(n)))
    assert np.isclose(Z2_mine, Z2_pt, rtol=1e-10), (Z2_mine, Z2_pt)
    # Z3 is symmetric and finite in the covariance configuration (k, -k, q)
    k1j, k2j, nj = jnp.array(k1), jnp.array(k2), jnp.array(n)
    z3a = float(kern.Z3(k1j, -k1j, k2j, nj)); z3b = float(kern.Z3(k2j, k1j, -k1j, nj))
    assert np.isfinite(z3a) and np.isclose(z3a, z3b)
    # approaching the squeezed configuration continuously (k3 -> -k2)
    z3_eps = [float(kern.Z3(k1j, k2j, -k2j * (1 + eps), nj)) for eps in (1e-2, 1e-3, 1e-4)]
    z3_0 = float(kern.Z3(k1j, k2j, -k2j, nj))
    assert np.isfinite(z3_0) and abs(z3_eps[-1] - z3_0) < 1e-2 * abs(z3_0) + 1e-6, (z3_eps, z3_0)
    # real space, matter: Z1 = 1, Z2 = F2, Z3 = F3
    kern0 = tj._PTKernels(0., dict(b1=1., b2=0., bG2=0., b3=0., bG3=0., bdG2=0., bGamma3=0.))
    k3 = rng.normal(size=3) * 0.1
    assert np.isclose(float(kern0.Z2(k1j, k2j, nj)), F2)
    assert np.isclose(float(kern0.Z3(k1j, k2j, jnp.array(k3), nj)), float(kern0.F3(k1j, k2j, jnp.array(k3))))


def test_trispectrum_real_space_matter():
    """Real-space matter T0 for the monopole against a brute-force quadrature of the parallelogram trispectrum (both channels explicitly)."""
    pk_lin, poles, f, b1 = get_pk()
    bias = dict(b1=1., b2=0., bG2=0., b3=0., bG3=0., bdG2=0., bGamma3=0.)
    k = np.array([0.05, 0.1])
    V = 1e9
    mine = tj.compute_trispectrum_covariance(k, pk_lin, 0., bias, volume=V, include_shotnoise=False, ells=(0,), nq=96, ntheta=4, nphi=4)[0, 0]
    kern = tj._PTKernels(0., bias)
    x, w = np.polynomial.legendre.leggauss(4000)
    n = jnp.array([0., 0., 1.])
    pkj = tj._log_interpolator(*tj._tabulate_pk(pk_lin))

    def T(k1, k2):
        mu = jnp.array(x)
        k1v = jnp.stack([0 * mu, 0 * mu, 0 * mu + k1], -1)
        k2v = jnp.stack([k2 * jnp.sqrt(1 - mu**2), 0 * mu, k2 * mu], -1)
        out = 0.
        for sign in (1., -1.):
            qv = k1v + sign * k2v
            q = jnp.linalg.norm(qv, axis=-1)
            Z1 = kern.Z2(-k1v, qv, n); Z2 = kern.Z2(-sign * k2v, qv, n)
            out = out + pkj(q) * (4 * pkj(k1)**2 * Z1**2 + 4 * pkj(k2)**2 * Z2**2 + 8 * pkj(k1) * pkj(k2) * Z1 * Z2)
        out = out + 12 * (pkj(k1)**2 * pkj(k2) * kern.Z3(k1v, -k1v, k2v, n) + pkj(k2)**2 * pkj(k1) * kern.Z3(k2v, -k2v, k1v, n))
        return float(jnp.sum(jnp.array(w) * out)) / 2 / V
    for i, j in [(0, 1), (1, 1)]:
        ref = T(k[i], k[j])
        assert np.isclose(mine[i, j], ref, rtol=3e-3), (i, j, mine[i, j], ref)


def test_exact_high_order_products():
    """Exact L2 L4 / L4 L4 window products: identical (0,0) kernels, small changes elsewhere; contraction helper vs brute force."""
    import itertools
    rng = np.random.default_rng(0)
    for na, nb in [(2, 4), (4, 2), (4, 4)]:
        n = na + nb; idx = tj._IDX[n]; W = rng.normal(size=(len(idx), 5))
        T = np.zeros((3,) * n + (5,))
        for m, t in enumerate(idx):
            for perm in set(itertools.permutations(t)): T[perm] = W[m]
        a, b = rng.normal(size=3), rng.normal(size=(3, 5))
        ref = T
        for _ in range(na): ref = np.tensordot(a, ref, axes=(0, 0))
        for _ in range(nb): ref = np.einsum('i...k,ik->...k', ref, b)
        assert np.allclose(np.asarray(tj._contract_split(jnp.array(W), a, jnp.array(b), na, nb)), ref, rtol=1e-10)
    L, N = 1500., 50000
    pos = rng.uniform(-L / 2, L / 2, size=(N, 3)); pos[:, 2] += 20 * L
    win = tj.SurveyWindow(pos, nz=np.full(N, 1e-4), alpha=0.1, kmax_window=0.02)
    kedges = tj.linear_kedges(0.02, 0.05, 0.01)
    kA = tj.compute_window_kernels(win, kedges, kmodes_sampled=8, batch_size=4, progress=False, seed=1, exact_high_order=False)
    kE = tj.compute_window_kernels(win, kedges, kmodes_sampled=8, batch_size=4, progress=False, seed=1, exact_high_order=True)
    assert np.array_equal(kE[..., 0], kA[..., 0])                      # (0,0) does not involve the products
    assert np.abs(kE[..., 3] - kA[..., 3]).max() < 1e-4 * np.abs(kA[..., 3]).max()   # (2,0) only through the k1 <-> k2 sums
    assert 1e-3 < np.abs(kE[..., 2] - kA[..., 2]).max() / np.abs(kA[..., 2]).max() < 0.5   # (4,4) changes


def test_mesh_window():
    """MeshWindow.from_randoms (W22 = W11_A W11_B on a fine mesh, two random halves) vs SurveyWindow (catalogue sum) for uniform weights."""
    rng = np.random.default_rng(3)
    L, N, nbar = 1500., 1_000_000, 1e-4
    pos = rng.uniform(-L / 2, L / 2, size=(N, 3)); pos[:, 2] += 20 * L
    alpha = nbar * L**3 / N
    win = tj.SurveyWindow(pos, nz=np.full(N, nbar), alpha=alpha, kmax_window=0.02)
    mw = tj.MeshWindow.from_randoms(pos, np.full(N, alpha), w12=np.full(N, alpha), boxsize=win.boxsize, boxcenter=np.asarray(win.attrs.boxcenter),
                                    fine_meshsize=4 * win.meshsize, meshsize=win.meshsize)
    # for a uniform cube I22 = nbar^2 L^3; the cross product of two random halves is free of the randoms' Poisson noise
    # (squaring one painted catalogue gives 1.5 x more with ~2 randoms per cell); the residual is the mesh smoothing of the edges
    assert np.isclose(mw.I('22'), nbar**2 * L**3, rtol=0.03), (mw.I('22'), nbar**2 * L**3)
    assert np.isclose(mw.I('12'), win.I('12'), rtol=1e-6)
    a, b = np.asarray(mw.W('W22')), np.asarray(win.W('W22'))
    assert np.isclose(a[0, 0, 0].real / b[0, 0, 0].real, mw.I('22') / win.I('22'), rtol=1e-6)
    # normalized window shapes agree at low q; at larger q the catalogue noise of the cross product shows (see VALIDATION.md, section 7)
    ik = win.ikgrid(); kr = np.sqrt((ik**2).sum(0)) * win.kfun
    sel = (kr > 0) & (kr < 0.004)
    ratio = (a / a[0, 0, 0]) / (b / b[0, 0, 0])
    assert np.median(np.abs(ratio[sel] - 1)) < 0.03, np.median(np.abs(ratio[sel] - 1))
    kedges = tj.linear_kedges(0.02, 0.05, 0.01)
    kA = tj.compute_window_kernels(win, kedges, kmodes_sampled=16, batch_size=4, progress=False, seed=1, exact_high_order=False)
    kB = tj.compute_window_kernels(mw, kedges, kmodes_sampled=16, batch_size=4, progress=False, seed=1, exact_high_order=False)
    d = np.diag(kB[..., 0, 0]) / np.diag(kA[..., 0, 0]) * (win.I('22') / mw.I('22'))**2
    assert np.all(np.abs(d - 1) < 0.05), d


def test_cube_vs_box(L=1500., nbar=1e-4, nrandoms=int(4e5), kmodes=200):
    """A cube of randoms far from the observer must reproduce the periodic-box Gaussian covariance up to the window leakage."""
    pk_lin, poles, f, b1 = get_pk()
    rng = np.random.default_rng(42)
    pos = rng.uniform(-L / 2, L / 2, size=(nrandoms, 3)); pos[:, 2] += 20 * L
    alpha = nbar * L**3 / nrandoms
    win = tj.SurveyWindow(pos, nz=np.full(nrandoms, nbar), alpha=alpha, kmax_window=0.02)
    assert np.isclose(win.I('22'), nbar**2 * L**3, rtol=1e-6) and np.isclose(win.I('12'), nbar * L**3, rtol=1e-6)
    kedges = tj.linear_kedges(0.02, 0.12, 0.01)
    k = kedges[:-1] + 0.005
    pk = poles(k)
    t0 = time.time()
    kernels = tj.compute_window_kernels(win, kedges, kmodes_sampled=kmodes, batch_size=8, seed=1, progress=False, exact_high_order=False)
    print(f'kernels: {time.time() - t0:.0f}s on {jax.default_backend()}')
    survey = tj.gaussian_covariance_survey(kernels, pk, alpha=alpha, I22=win.I('22'))
    box = tj.gaussian_covariance_box(pk, tj.count_modes(kedges, L), shotnoise=(1 + alpha) / nbar)
    for (i, j) in [(0, 0), (1, 1), (2, 2)]:
        diag = np.diag(survey[i, j]) / np.diag(box[i, j])
        rowsum = survey[i, j].sum(axis=1) / np.diag(box[i, j])
        print(f'  ell={2*i}: diag ratio', np.round(diag, 3), 'row-sum ratio', np.round(rowsum, 3))
        # leakage into neighbours lowers the diagonal (~0.78 for L=1500), the row sums conserve the total up to the
        # truncated sinc^2 tails (6 / (pi kmax_window L) ~ 6%) and the Monte-Carlo noise
        # the quadrupole kernels are noisier for a given number of sampled modes
        lo, hi = (0.85, 1.1) if i == 0 else (0.7, 1.15)
        assert np.all((diag > 0.55) & (diag < 1.0))
        assert np.all((rowsum[1:-1] > lo) & (rowsum[1:-1] < hi))


if __name__ == '__main__':
    slow = '--slow' in sys.argv
    for name, func in list(globals().items()):
        if name.startswith('test_') and callable(func):
            if name == 'test_cube_vs_box' and not slow:
                continue
            t0 = time.time()
            func()
            print(f'{name}: ok ({time.time() - t0:.1f}s)')
