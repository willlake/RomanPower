r"""
Analytic covariance of power spectrum multipoles :math:`P_\ell(k)`, :math:`\ell = 0, 2, 4`.

This is a JAX port of ``thecov`` (Alves et al., https://github.com/cosmodesi/thecov), itself an
implementation of CovaPT (Wadekar & Scoccimarro 2020, arXiv:1910.02914) and of the FFTLog trispectrum of
Kobayashi (2023, arXiv:2308.08593), rebuilt on top of the RomanPower mesh / painting infrastructure:

* **Gaussian term with the survey window** (:func:`compute_window_kernels`, :func:`gaussian_covariance_survey`):
  Eqs. (57) and (88) of Wadekar & Scoccimarro. The window functions :math:`W_{22}(x) = \bar n^2 w^2` and
  :math:`W_{12}(x) = \bar n w^2` are painted from the random catalogue together with their Cartesian moments
  :math:`W(x) \hat x_i \hat x_j` and :math:`W(x) \hat x_i \hat x_j \hat x_k \hat x_l`, FFT'd, and contracted
  with :math:`\hat k_1`, :math:`\hat k_2` to give the Legendre multipoles of the window with respect to the two
  wavevectors. For every sampled mode :math:`k_1` the products of window multipoles are summed over all
  :math:`k_2 = k_1 - q` of the FFT grid, binned by the :math:`k_2` shell.
  Unlike CovaPT, the products of window multipoles :math:`L_2 L_4`, :math:`L_4 L_4` are exact by default (6th and
  8th order moments are painted); CovaPT's approximation is available with ``exact_high_order=False``.
* **Regular trispectrum** :math:`T_0` (:func:`compute_trispectrum_covariance`): Eq. (7) of Kobayashi (2023)
  (Eq. (66) of Wadekar & Scoccimarro for the survey factors :math:`V \to I_{22}^2 / I_{44}`), plus the shot-noise
  trispectrum of Appendix C (:math:`1/\bar n \to I_{34} / I_{44}`, :math:`1/\bar n^2 \to I_{24} / I_{44}`).
  Unlike ``thecov`` (which delegates to ``powercovfft``, i.e. Mathematica-generated angle-integrated
  expressions + FFTLog master integrals), the redshift-space kernels :math:`Z_1, Z_2, Z_3` are written down
  explicitly (Appendix A of Kobayashi 2023, with the third-order bias operators) and the angular integrals
  are done numerically: the line-of-sight average is a polynomial of finite degree in :math:`\hat n` and is
  integrated exactly with a Gauss-Legendre :math:`\times` uniform-azimuth product rule; the remaining integral over
  :math:`\hat k_1 \cdot \hat k_2` is done in the variable :math:`q = |k_1 + k_2|` (log-spaced Gauss-Legendre),
  which resolves the :math:`P(q) / q^2` structure of the "snake" term near :math:`k_1 = k_2`.
* **Super-sample covariance** (:func:`compute_ssc_covariance`): beat-coupling and local-average terms,
  Eqs. (64)-(65) and (77)-(78) of Wadekar & Scoccimarro, following CovaPT's ``Covariance_SurveyGeometry``
  notebook with the corrections applied in the local ``thecov`` fork (galaxy :math:`P_\mathrm{lin} = b_1^2 P_m`,
  Kaiser factor and Poisson term in :math:`\langle \delta n_g^2 \rangle`, :math:`2/I_{10}` term).

All heavy lifting (painting, FFTs, kernel sums, trispectrum quadrature) runs in JAX and is GPU-friendly.
Double precision is required: the module switches ``jax_enable_x64`` on at import.

Conventions
-----------
Multipole blocks are returned as arrays of shape ``(nells, nells, nk, nk)``; :func:`blocks_to_matrix` stacks
them into the ``(nells * nk, nells * nk)`` matrix ordered as ``[P_0(k), P_2(k), P_4(k)]`` (the layout of
``thecov.base.MultipoleCovariance.cov``).

Window weights follow ``thecov``: :math:`W_{ij} = \alpha\, \bar n^{i-1} w_\mathrm{FKP}^j w_c` where
:math:`\bar n` is the ``nz`` column, :math:`w_c` the completeness weight and :math:`\alpha = N_\mathrm{data} /
N_\mathrm{randoms}`; any window can be overridden through ``windows={'W12': ...}`` (e.g. to use
:math:`w_c^2` in the shot-noise window, see the notes in ``thecov/claude.md``).
"""

import itertools
import logging
import math
import warnings

import numpy as np
import jax
from jax import numpy as jnp

jax.config.update('jax_enable_x64', True)

from .mesh import MeshAttrs, ParticleField, _bincount, staticarray
from .utils import get_legendre


logger = logging.getLogger('thecov')

__all__ = ['SurveyWindow', 'MeshWindow', 'count_kbins', 'linear_kedges', 'sample_kmodes', 'count_modes',
           'compute_window_kernels', 'gaussian_covariance_survey', 'gaussian_covariance_box',
           'compute_trispectrum_covariance', 'compute_ssc_covariance', 'pbs_bias_parameters',
           'blocks_to_matrix', 'matrix_to_blocks', 'to_covariance_matrix', 'kernel_labels']


# ----------------------------------------------------------------------------------------------------------------------
# k-binning helpers
# ----------------------------------------------------------------------------------------------------------------------

def count_kbins(kmin, kmax, dk):
    """Number of linear k-bins in ``[kmin, kmax)``, robust to floating-point round-off (as ``thecov.math.count_kbins``)."""
    n = (kmax - kmin) / dk
    return int(np.round(n)) if np.isclose(n, np.round(n), rtol=0., atol=1e-6) else int(np.ceil(n))


def linear_kedges(kmin, kmax, dk):
    """Edges of the linear k-bins ``kmin + dk * [0, 1, ..., nbins]``."""
    return kmin + dk * np.arange(count_kbins(kmin, kmax, dk) + 1)


def _check_linear_edges(kedges):
    kedges = np.asarray(kedges, dtype=float)
    dk = np.diff(kedges)
    if not np.allclose(dk, dk[0], rtol=1e-6, atol=0.):
        raise ValueError('window kernels require linear k-bins')
    dk = float(dk.mean())
    kmin = float(kedges[0])
    if not np.isclose(kmin / dk, np.round(kmin / dk), rtol=0., atol=1e-6):
        warnings.warn(f'kmin = {kmin} is not an integer multiple of dk = {dk}: the k2-bin assignment of the window '
                      'kernels assumes bin edges at integer multiples of dk')
    return kedges, kmin, dk


def _leggauss(n, a=-1., b=1.):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _legendre_np(ell):
    return {0: lambda x: np.ones_like(x), 2: lambda x: (3 * x**2 - 1) / 2., 4: lambda x: (35 * x**4 - 30 * x**2 + 3) / 8.,
            6: lambda x: (231 * x**6 - 315 * x**4 + 105 * x**2 - 5) / 16.}[ell]


# ----------------------------------------------------------------------------------------------------------------------
# Sampling of k1 modes (same semantics as thecov.math.sample_kmodes, with the local fixes)
# ----------------------------------------------------------------------------------------------------------------------

def count_modes(kedges, boxsize, exact_below=None):
    r"""
    Number of FFT modes of a periodic box of side ``boxsize`` in each shell of ``kedges``.

    Parameters
    ----------
    exact_below : float, default=None
        Shells below this wavenumber are counted exactly by enumerating the integer lattice, the others with the
        continuum :math:`V k^2 dk / (2 \pi^2)`. ``None`` counts all shells exactly, ``0`` none.
    """
    kedges = np.asarray(kedges, dtype=float)
    kfun = 2 * np.pi / boxsize
    nmodes = boxsize**3 / (6 * np.pi**2) * (kedges[1:]**3 - kedges[:-1]**3)
    kex = kedges[-1] if exact_below is None else min(exact_below, kedges[-1])
    iL = int(np.ceil(kex / kfun))
    if iL > 0:
        ix = np.arange(-iL, iL + 1)
        ir2 = (ix[:, None, None]**2 + ix[None, :, None]**2 + ix[None, None, :]**2).ravel()
        ir = np.sqrt(ir2) * kfun
        counts, _ = np.histogram(ir, bins=kedges)
        sel = kedges[1:] <= kex * (1 + 1e-9)
        nmodes[sel] = counts[sel]
    return nmodes


def sample_kmodes(kedges, boxsize, max_modes=1000, k_shell_approx=0.1, seed=None, rng=None):
    """
    Sample integer modes :math:`(i_x, i_y, i_z, |i|)` (in units of the fundamental wavenumber) in each k-shell.

    Below ``k_shell_approx`` all lattice modes of the shell are enumerated and (at most) ``max_modes`` of them are
    drawn (with replacement, as ``thecov``); above, ``max_modes`` modes are drawn uniformly in the continuous shell
    and rounded to the lattice. Returns the list of ``(n_i, 4)`` arrays and the number of modes per shell
    (exact counts below ``k_shell_approx``, continuum above), following ``thecov.math.sample_kmodes``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    kedges = np.asarray(kedges, dtype=float)
    kfun = 2 * np.pi / boxsize
    nbins = len(kedges) - 1
    i_shell = int(np.searchsorted(kedges, k_shell_approx * (1 + 1e-9), side='right') - 1)
    i_shell = min(max(i_shell, 0), nbins)
    modes, nmodes = [], []
    if i_shell > 0:
        iL = int(np.ceil(kedges[i_shell] / kfun))
        ix, iy, iz = np.mgrid[-iL:iL + 1, -iL:iL + 1, -iL:iL + 1]
        ir = np.sqrt(ix**2 + iy**2 + iz**2)
        sort = np.floor((ir * kfun - kedges[0]) / (kedges[1] - kedges[0]) + 1e-9).astype(int)
        for i in range(i_shell):
            mask = sort == i
            N = int(mask.sum())
            sel = np.array([ix[mask], iy[mask], iz[mask], ir[mask]]).T
            if N > max_modes:
                sel = sel[rng.integers(N, size=max_modes)]
            modes.append(sel)
            nmodes.append(N)
    for i in range(i_shell, nbins):
        rmin, rmax = kedges[i] / kfun, kedges[i + 1] / kfun
        out = []
        while len(out) < max_modes:
            n = 2 * (max_modes - len(out)) + 16
            r = rmin + (rmax - rmin) * rng.random(n)
            theta = 2 * np.pi * rng.random(n)
            phi = np.arccos(1 - 2 * rng.random(n))
            xyz = np.round(np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)]).T)
            rr = np.sqrt((xyz**2).sum(axis=1))
            ok = (rr >= rmin) & (rr <= rmax)
            out.extend(np.column_stack([xyz[ok], rr[ok]]))
        modes.append(np.array(out[:max_modes]))
        nmodes.append(boxsize**3 / (6 * np.pi**2) * (kedges[i + 1]**3 - kedges[i]**3))
    return modes, np.array(nmodes, dtype=float)


# ----------------------------------------------------------------------------------------------------------------------
# Survey window: painting of the random catalogue
# ----------------------------------------------------------------------------------------------------------------------

_XYZ = ('x', 'y', 'z')
_IDX2 = list(itertools.combinations_with_replacement(range(3), 2))   # xx, xy, xz, yy, yz, zz
_IDX4 = list(itertools.combinations_with_replacement(range(3), 4))   # xxxx, xxxy, ..., zzzz (15)
_LABELS2 = [''.join(_XYZ[i] for i in idx) for idx in _IDX2]
_LABELS4 = [''.join(_XYZ[i] for i in idx) for idx in _IDX4]
_IDX6 = list(itertools.combinations_with_replacement(range(3), 6))   # 28 moments
_IDX8 = list(itertools.combinations_with_replacement(range(3), 8))   # 45 moments
_LABELS6 = [''.join(_XYZ[i] for i in idx) for idx in _IDX6]
_LABELS8 = [''.join(_XYZ[i] for i in idx) for idx in _IDX8]
_IDX = {2: _IDX2, 4: _IDX4, 6: _IDX6, 8: _IDX8}
_LABELS = {2: _LABELS2, 4: _LABELS4, 6: _LABELS6, 8: _LABELS8}


def _contract_split(Wn, a, b, na, nb):
    r"""
    :math:`\sum W_{i_1 \ldots i_{n_a} j_1 \ldots j_{n_b}} a_{i_1} \cdots a_{i_{n_a}} b_{j_1} \cdots b_{j_{n_b}}` for a symmetric tensor
    of rank ``na + nb`` stored by sorted multiset (``Wn[m]``), with ``a`` scalars (or broadcastable) and ``b`` arrays:
    the ``a`` indices are summed first into a rank-``nb`` tensor of coefficients (:math:`3^{n_a}` terms per component).
    """
    pos = {t: m for m, t in enumerate(_IDX[na + nb])}
    out = 0.
    for jb in itertools.combinations_with_replacement(range(3), nb):
        mult = len(set(itertools.permutations(jb)))
        bprod = 1.
        for j in jb:
            bprod = bprod * b[j]
        coeff = 0.
        for ia in itertools.product(range(3), repeat=na):
            aprod = 1.
            for i in ia:
                aprod = aprod * a[i]
            coeff = coeff + Wn[pos[tuple(sorted(ia + jb))]] * aprod
        out = out + mult * coeff * bprod
    return out


def _contract2(W2, a, b):
    r"""Return :math:`\sum_{ij} W_{ij} a_i b_j` from the 6 independent moments ``W2[m]`` (symmetric tensor)."""
    toret = 0.
    for m, (i, j) in enumerate(_IDX2):
        if i == j:
            toret = toret + W2[m] * (a[i] * b[j])
        else:
            toret = toret + W2[m] * (a[i] * b[j] + a[j] * b[i])
    return toret


def _contract4(W4, a, b, c, d):
    r"""Return :math:`\sum_{ijkl} W_{ijkl} a_i b_j c_k d_l` from the 15 independent moments ``W4[m]``."""
    toret = 0.
    for m, idx in enumerate(_IDX4):
        coeff = 0.
        for perm in set(itertools.permutations(idx)):
            coeff = coeff + a[perm[0]] * b[perm[1]] * c[perm[2]] * d[perm[3]]
        toret = toret + W4[m] * coeff
    return toret


class SurveyWindow(object):
    r"""
    Survey window functions from a random catalogue, painted with the RomanPower mesh machinery.

    Parameters
    ----------
    positions : array (N, 3)
        Cartesian positions of the randoms, observer at the origin (so that the line of sight is :math:`\hat x`).
    nz : array (N,), optional
        Mean number density at the position of each random (the ``NZ`` column). Default 1.
    weight : array (N,), optional
        Completeness-type weight :math:`w_c` (``WEIGHT``). Default 1.
    weight_fkp : array (N,), optional
        FKP weight (``WEIGHT_FKP``). Default 1.
    alpha : float, default=1.
        :math:`N_\mathrm{data} / N_\mathrm{randoms}`, multiplies all windows.
    windows : dict, optional
        Overrides ``{'W12': array (N,), ...}`` for the per-random weights of given windows.
    kmax_window : float, default=0.02
        Nyquist wavenumber of the window FFT grid; sets the cell size :math:`\pi / k_\mathrm{max,window}` unless
        ``cellsize`` or ``meshsize`` is given. The window kernels only include :math:`|k_1 - k_2|` up to this value.
    meshsize, cellsize, boxsize, boxcenter, boxpad :
        Mesh geometry, as in :func:`romanpower.get_mesh_attrs` (box = ``boxpad`` times the extent of the randoms).
    resampler, interlacing, compensate :
        Painting options (default TSC, interlacing 3, compensated, as ``thecov`` / CovaPT).
    chunk_size : int, default=4e6
        Randoms are painted in chunks of this size (the painting is linear in the weights).
    """
    logger = logging.getLogger('SurveyWindow')

    def __init__(self, positions, nz=None, weight=None, weight_fkp=None, alpha=1., windows=None, kmax_window=0.02,
                 meshsize=None, cellsize=None, boxsize=None, boxcenter=None, boxpad=2., resampler='tsc', interlacing=3,
                 compensate=True, chunk_size=4_000_000):
        positions = np.asarray(positions, dtype='f8')
        self.size = len(positions)
        self.positions = positions
        self.nz = np.ones(self.size) if nz is None else np.asarray(nz, dtype='f8')
        self.weight = np.ones(self.size) if weight is None else np.asarray(weight, dtype='f8')
        self.weight_fkp = np.ones(self.size) if weight_fkp is None else np.asarray(weight_fkp, dtype='f8')
        self.alpha = float(alpha)
        self._windows = dict(windows or {})
        self.kmax_window = kmax_window
        self.paint_kwargs = dict(resampler=resampler, interlacing=interlacing, compensate=compensate)
        self.chunk_size = int(chunk_size)

        pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
        if boxcenter is None:
            boxcenter = 0.5 * (pos_min + pos_max)
        boxcenter = np.asarray(boxcenter, dtype='f8') * np.ones(3)
        if boxsize is None and meshsize is None:
            if cellsize is None:
                cellsize = np.pi / kmax_window / (1. + 1e-9)
            boxsize = np.max(pos_max - pos_min) * boxpad
            meshsize = int(np.ceil(boxsize / cellsize))
            meshsize += meshsize % 2  # even, as CovaPT recommends
            boxsize = meshsize * cellsize
        elif meshsize is None:
            if cellsize is None:
                cellsize = np.pi / kmax_window / (1. + 1e-9)
            meshsize = int(np.ceil(np.max(boxsize) / cellsize))
            meshsize += meshsize % 2
        elif boxsize is None:
            boxsize = np.max(pos_max - pos_min) * boxpad if cellsize is None else meshsize * cellsize
        boxsize = float(np.max(boxsize))
        meshsize = int(np.max(meshsize))
        # complex dtype: full (non-Hermitian) FFT cube, as the kernels need W(q) for all q = k1 - k2
        self.attrs = MeshAttrs(meshsize=meshsize, boxsize=boxsize, boxcenter=boxcenter, dtype=complex, fft_backend='jax')
        self.boxsize, self.meshsize = boxsize, meshsize
        self._W = {}
        self._I = {}
        self._window_power = None
        self.logger.info(f'Box size {boxsize:.1f}, mesh size {meshsize}, cell size {self.cellsize:.2f}, '
                         f'kfun = {self.kfun:.5f}, knyquist = {self.knyquist:.4f}; {self.size} randoms.')

    # --- geometry
    @property
    def cellsize(self):
        return self.boxsize / self.meshsize

    @property
    def kfun(self):
        return 2 * np.pi / self.boxsize

    @property
    def knyquist(self):
        return np.pi * self.meshsize / self.boxsize

    @property
    def delta_k_max(self):
        """Number of k-bins on each side of the diagonal covered by the window FFT grid, as ``thecov`` (``nmesh // 2 - 1``)."""
        return self.meshsize // 2 - 1

    def ikgrid(self):
        """Integer wavenumber grid ``(3, n, n, n)`` in FFT order."""
        ik = np.fft.fftfreq(self.meshsize, d=1. / self.meshsize)
        return np.array(np.meshgrid(ik, ik, ik, indexing='ij'))

    # --- per-random weights of the windows
    @staticmethod
    def _parse(label):
        w = label.lower().replace('w', '').replace('i', '')
        base = ''.join(c for c in w if c.isdigit())
        moment = ''.join(c for c in w if c in _XYZ)
        assert len(base) == 2, f'invalid window label {label}'
        return base, moment

    def weights(self, label):
        r"""Per-random weight of window ``label`` ('W22', 'W12', 'W10', 'W22xx', ...): :math:`\alpha \bar n^{i-1} w_\mathrm{FKP}^j w_c \hat x_a \hat x_b ...`."""
        base, moment = self._parse(label)
        key = 'W' + base
        if key in self._windows:
            w = np.asarray(self._windows[key], dtype='f8')
        else:
            i, j = int(base[0]), int(base[1])
            w = self.alpha * self.nz**(i - 1) * self.weight_fkp**j * self.weight
        if moment:
            x = self.positions
            r2 = np.sum(x**2, axis=-1)
            xhat = x / np.sqrt(r2)[:, None]
            for c in moment:
                w = w * xhat[:, _XYZ.index(c)]
        return w

    def I(self, label):
        r"""Integral :math:`I_{ij} = \sum \alpha \bar n^{i-1} w_\mathrm{FKP}^j w_c` (``I22``, ``I12``, ...)."""
        base, _ = self._parse(label)
        if base not in self._I:
            self._I[base] = float(self.weights('W' + base).sum())
        return self._I[base]

    def sum_weights_product(self, label1, label2):
        """Sum over randoms of the product of two window weights (Poisson term of a window power spectrum)."""
        return float(np.sum(self.weights(label1) * self.weights(label2)))

    # --- painting
    def paint(self, weights):
        """Paint per-random ``weights`` to the mesh and return the (full, complex) FFT, normalized so that W(k=0) = sum(weights)."""
        out = None
        for start in range(0, self.size, self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            mesh = ParticleField(self.positions[sl], weights[sl], attrs=self.attrs).paint(out='complex', **self.paint_kwargs)
            out = mesh.value if out is None else out + mesh.value
        return out

    def W(self, label):
        """FFT of window ``label`` (cached), array ``(n, n, n)`` in FFT order."""
        base, moment = self._parse(label)
        key = 'W' + base + moment
        if key not in self._W:
            self._W[key] = self.paint(self.weights(key))
        return self._W[key]

    def compute_window_ffts(self, base='22', orders=(0, 2, 4)):
        """
        Compute (and cache) the Cartesian moments of window ``base`` for the given even ``orders`` (up to 8);
        return ``(W0, W2 (6, ...), W4 (15, ...)[, W6 (28, ...), W8 (45, ...)])`` (``None`` for orders not requested).
        """
        base = self._parse(base)[0]
        moments = [''] * (0 in orders) + sum((_LABELS[n] for n in (2, 4, 6, 8) if n in orders), [])
        for m in moments:
            self.W('W' + base + m)
        out = [self._W.get('W' + base)]
        for n in (2, 4, 6, 8):
            if n in orders:
                out.append(jnp.array([self._W['W' + base + m] for m in _LABELS[n]]))
            elif n <= max(orders):
                out.append(None)
        return tuple(out)

    def clear(self):
        self._W = {}
        self._I = {}
        self._window_power = None

    # --- window power spectra for the SSC term
    def multipole_fields(self, base='22'):
        r"""Legendre multipoles :math:`W_{L}(q) = \int d^3x W(x) L_L(\hat q \cdot \hat x) e^{-i q \cdot x}`, L = 0, 2, 4, on the mesh."""
        W0, W2, W4 = self.compute_window_ffts(base)
        ik = jnp.array(self.ikgrid())
        ikr = jnp.sqrt(jnp.sum(ik**2, axis=0))
        khat = jnp.where(ikr > 0, ik / jnp.where(ikr > 0, ikr, 1.), 0.)
        L0 = W0
        L2 = 1.5 * _contract2(W2, khat, khat) - 0.5 * L0
        L4 = 35. / 8. * _contract4(W4, khat, khat, khat, khat) - 5. / 2. * L2 - 7. / 8. * L0
        return L0, L2, L4

    def compute_window_power(self, kedges=None, bases=('22', '10'), return_all=False):
        r"""
        Window power spectra needed by the super-sample covariance, normalized as in CovaPT:
        :math:`P_{ab}(k) = (2 L_a + 1)(2 L_b + 1) \langle W_a(k) W_b^*(k) \rangle_{|k|} / (I_a I_b)`,
        with the Poisson (randoms) shot noise removed from the :math:`L=0 \times L=0` spectra, for all 21 pairs of
        ``(W22_L0, W22_L2, W22_L4, W10_L0, W10_L2, W10_L4)`` (``itertools.combinations_with_replacement``
        order, i.e. the first 16 are those of ``thecov``).

        Returns
        -------
        kavg : array
            Mean :math:`|k|` in each bin.
        power : array (21, nbins)
        """
        if self._window_power is not None:
            return self._window_power
        if kedges is None:
            kedges = np.arange(0., self.knyquist + self.kfun / 2, self.kfun)
        kedges = np.asarray(kedges)
        nbins = len(kedges) - 1
        ik = self.ikgrid()
        kr = np.sqrt(np.sum(ik**2, axis=0)) * self.kfun
        # _bincount follows jnp.digitize: index 0 (below the first edge) and nbins + 1 (above the last) are dropped
        ibin = jnp.array(np.digitize(kr, kedges))
        counts = np.asarray(_bincount(ibin, jnp.ones(ibin.shape), length=nbins))
        kavg = np.asarray(_bincount(ibin, jnp.array(kr), length=nbins)) / np.where(counts > 0, counts, 1)

        fields, norms, labels = [], [], []
        for base in bases:
            L0, L2, L4 = self.multipole_fields(base)
            fields += [L0, L2, L4]
            norms += [self.I(base)] * 3
            labels += [(base, 0), (base, 2), (base, 4)]
        power = []
        for (a, b) in itertools.combinations_with_replacement(range(len(fields)), 2):
            pab = np.asarray(_bincount(ibin, (fields[a] * jnp.conj(fields[b])).real, length=nbins)) / np.where(counts > 0, counts, 1)
            (base_a, ell_a), (base_b, ell_b) = labels[a], labels[b]
            if ell_a == 0 and ell_b == 0:
                pab -= self.sum_weights_product('W' + base_a, 'W' + base_b)  # Poisson shot noise of the discrete randoms
            pab *= (2 * ell_a + 1) * (2 * ell_b + 1) / (norms[a] * norms[b])
            pab[0] = 1. if ell_a == ell_b else 0.  # k = 0 bin set by hand
            power.append(pab)
        power = np.array(power)
        if np.isnan(kavg[0]): kavg[0] = 0.
        self._window_power = (kavg, power)
        return self._window_power


class MeshWindow(SurveyWindow):
    r"""
    Survey window given as meshes instead of a random catalogue: :math:`W_{22}(x)` (and optionally :math:`W_{12}(x)`)
    as arrays of *weights per cell* (i.e. :math:`\int_\mathrm{cell} W(x) d^3x`) on a (fine) mesh, whose Cartesian
    moments are FFT'd and cut to the ``meshsize`` cube of the kernel computation. Meant for tests with Gaussian random
    fields multiplied by a known window :math:`W_{11}(x)`: then :math:`W_{22} = W_{11}^2 / V_\mathrm{cell}` and
    :math:`W_{12} = 0` (no shot noise).

    Parameters
    ----------
    W22 : array (n, n, n)
        Weights per cell of :math:`W_{22}` on the fine mesh (real).
    attrs : MeshAttrs
        Attributes of the fine mesh (box size, box center, mesh size).
    meshsize : int
        Size of the FFT cube kept for the kernels (its Nyquist wavenumber is ``kmax_window``); must be <= the fine mesh.
    W12 : array, optional
        Weights per cell of :math:`W_{12}` (default 0).
    """
    def __init__(self, W22, attrs, meshsize, W12=None):
        self.fine_attrs = attrs
        self.fine_meshsize = int(attrs.meshsize[0])
        self.boxsize = float(attrs.boxsize[0])
        self.meshsize = int(meshsize)
        assert self.meshsize <= self.fine_meshsize
        self.kmax_window = np.pi * self.meshsize / self.boxsize
        self.attrs = MeshAttrs(meshsize=self.meshsize, boxsize=self.boxsize, boxcenter=attrs.boxcenter, dtype=complex, fft_backend='jax')
        self._cells = {'22': jnp.asarray(W22), '12': None if W12 is None else jnp.asarray(W12)}
        self.alpha = 1.
        self.size = 0
        self.paint_kwargs = {}
        self._W, self._I, self._window_power, self._windows = {}, {}, None, {}
        x = self.fine_attrs.rcoords(kind='position', sparse=True)
        r2 = sum(xx**2 for xx in x)
        self._xhat = [xx / jnp.sqrt(r2) for xx in x]
        n, m = self.fine_meshsize, self.meshsize
        self._cut = np.r_[0:m // 2, n - m // 2:n]  # FFT-order indices of the central cube |ik| < m/2
        self.logger.info(f'Mesh window: fine mesh {n}, kernel cube {m}, box size {self.boxsize:.1f}, knyquist {self.knyquist:.4f}.')

    @classmethod
    def from_randoms(cls, positions, w11, w12=None, boxsize=None, boxcenter=None, fine_meshsize=288, meshsize=36,
                     resampler='tsc', interlacing=3, compensate=True, chunk_size=4_000_000, split_seed=42):
        r"""
        Build the window from a random catalogue with :math:`W_{22} = W_{11}^2 / V_\mathrm{cell}` where :math:`W_{11}(x)`
        is the painted weighted density :math:`\alpha n_r w_c w_\mathrm{FKP}` (per-random weights ``w11``), i.e. the
        window of the FKP field, and :math:`W_{12}` painted from ``w12`` (e.g. :math:`\alpha (w_c w_\mathrm{FKP})^2`).
        This is the correct :math:`W_{22}` when the completeness weights vary; :class:`SurveyWindow`'s catalogue sum
        :math:`\alpha \sum_r n(z_r) w_c w_\mathrm{FKP}^2` (thecov's prescription) has one power of :math:`w_c` less
        unless the ``nz`` column is the weighted density (VALIDATION.md, section 7).

        The square is taken as the product of two independent random halves of the catalogue (each with doubled
        weights), so that the Poisson noise of the randoms (self-pairs) does not add a white floor to :math:`W_{22}(q)`
        (as ``split`` in :func:`romanpower.cov2.compute_fkp2_covariance_window`); ``split_seed`` draws the halves.
        """
        positions = np.asarray(positions, dtype='f8')
        if boxcenter is None or boxsize is None:
            pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
            if boxcenter is None: boxcenter = 0.5 * (pos_min + pos_max)
            if boxsize is None: boxsize = 2. * np.max(pos_max - pos_min)
        attrs = MeshAttrs(meshsize=int(fine_meshsize), boxsize=float(boxsize), boxcenter=np.asarray(boxcenter, dtype='f8') * np.ones(3), dtype=float, fft_backend='jax')
        kw = dict(resampler=resampler, interlacing=interlacing, compensate=compensate, out='real')

        def paint(w):
            out = None
            for start in range(0, len(positions), int(chunk_size)):
                sl = slice(start, start + int(chunk_size))
                m = ParticleField(positions[sl], np.asarray(w)[sl], attrs=attrs).paint(**kw)
                out = m if out is None else out + m
            return out.value

        rng = np.random.default_rng(split_seed)
        half = rng.random(len(positions)) < 0.5
        w11 = np.asarray(w11, dtype='f8')
        W22 = paint(2. * w11 * half) * paint(2. * w11 * (~half)) / float(attrs.cellsize.prod())
        W12 = None if w12 is None else paint(w12)
        return cls(W22, attrs, meshsize, W12=W12)

    def _cells_of(self, base):
        c = self._cells.get(base)
        if c is None:
            c = jnp.zeros((self.fine_meshsize,) * 3)
        return c

    def weights(self, label):
        raise NotImplementedError('MeshWindow has no particles')

    def I(self, label):
        base, _ = self._parse(label)
        if base not in self._I:
            self._I[base] = float(jnp.sum(self._cells_of(base)))
        return self._I[base]

    def sum_weights_product(self, label1, label2):
        return 0.

    def W(self, label):
        base, moment = self._parse(label)
        key = 'W' + base + moment
        if key not in self._W:
            field = self._cells_of(base)
            for c in moment:
                field = field * self._xhat[_XYZ.index(c)]
            full = jnp.fft.fftn(field)
            self._W[key] = full[np.ix_(self._cut, self._cut, self._cut)]
        return self._W[key]


# ----------------------------------------------------------------------------------------------------------------------
# Gaussian window kernels (CovaPT Eqs. 57 & 88)
# ----------------------------------------------------------------------------------------------------------------------

#: Meaning of the 15 "terms" (axis -2 of the kernels), i.e. the power spectrum products they multiply (as CovaPT):
#: index 0-8: P_{l'}(k1) P_{l''}(k2) with (l', l'') in row-major order over (0, 2, 4);
#: 9-13: (1+alpha) times (P0(k1)+P0(k2))/2, P2(k1), P4(k1), P2(k2), P4(k2); 14: (1+alpha)^2.
kernel_labels = ['P0P0', 'P0P2', 'P0P4', 'P2P0', 'P2P2', 'P2P4', 'P4P0', 'P4P2', 'P4P4',
                 'P0', 'P2(k1)', 'P4(k1)', 'P2(k2)', 'P4(k2)', '1']
#: Multipole pairs of axis -1 of the kernels.
kernel_ells = [(0, 0), (2, 2), (4, 4), (2, 0), (4, 0), (4, 2)]


def _window_multipoles(W0, W2, W4, k1h, k2h, W6=None, W8=None):
    """
    Legendre multipoles of a window FFT with respect to k1 (scalars k1h) and k2 (grids k2h), as CovaPT.
    With ``W6`` and ``W8`` (6th and 8th order moments) the products L2(k1) L4(k2), L4(k1) L2(k2), L4(k1) L4(k2) are
    exact; otherwise they are CovaPT's approximation (k2 -> k1, L6 and L8 dropped).
    """
    xx = _contract2(W2, k1h, k1h)
    yy = _contract2(W2, k2h, k2h)
    L0 = W0
    k1L2 = 1.5 * xx - 0.5 * W0
    k2L2 = 1.5 * yy - 0.5 * W0
    x4 = _contract4(W4, k1h, k1h, k1h, k1h)
    y4 = _contract4(W4, k2h, k2h, k2h, k2h)
    k1L4 = 35. / 8. * x4 - 5. / 2. * k1L2 - 7. / 8. * W0
    k2L4 = 35. / 8. * y4 - 5. / 2. * k2L2 - 7. / 8. * W0
    k1k2 = _contract4(W4, k1h, k1h, k2h, k2h)  # x^2 y^2
    k1L2_k2L2 = 9. / 4. * k1k2 - 3. / 4. * xx - 1. / 2. * k2L2
    if W6 is not None and W8 is not None:
        x2y4 = _contract_split(W6, k1h, k2h, 2, 4)
        x4y2 = _contract_split(W6, k1h, k2h, 4, 2)
        x4y4 = _contract_split(W8, k1h, k2h, 4, 4)
        # L2(x) L4(y) = (3/2 x^2 - 1/2)(35/8 y^4 - 15/4 y^2 + 3/8)
        k1L2_k2L4 = 105. / 16. * x2y4 - 45. / 8. * k1k2 + 9. / 16. * xx - 35. / 16. * y4 + 15. / 8. * yy - 3. / 16. * W0
        k1L4_k2L2 = 105. / 16. * x4y2 - 45. / 8. * k1k2 + 9. / 16. * yy - 35. / 16. * x4 + 15. / 8. * xx - 3. / 16. * W0
        # L4(x) L4(y) = (35/8 x^4 - 15/4 x^2 + 3/8)(35/8 y^4 - 15/4 y^2 + 3/8)
        k1L4_k2L4 = (1225. / 64. * x4y4 - 525. / 32. * (x4y2 + x2y4) + 105. / 64. * (x4 + y4) + 225. / 16. * k1k2
                     - 45. / 32. * (xx + yy) + 9. / 64. * W0)
    else:
        # Products L2(k1) L4(k2), L4(k1) L2(k2), L4(k1) L4(k2) approximated by setting k2 -> k1 and dropping L6, L8
        # (6th and 8th order moments are not painted), as CovaPT
        k1L2_k2L4 = 2. / 7. * k1L2 + 20. / 77. * k1L4
        k1L4_k2L2 = k1L2_k2L4
        k1L4_k2L4 = 1. / 9. * L0 + 100. / 693. * k1L2 + 162. / 1001. * k1L4
    return dict(L0=L0, k1L2=k1L2, k2L2=k2L2, k1L4=k1L4, k2L4=k2L4, k1L2_k2L2=k1L2_k2L2,
                k1L2_k2L4=k1L2_k2L4, k1L4_k2L2=k1L4_k2L2, k1L4_k2L4=k1L4_k2L4)


def _kernel_terms(W, S, covapt_bugs=False):
    """
    The 15 x 6 window-kernel integrands of CovaPT (``Survey_window_kernels.ipynb``, ``WinFun``).

    ``W``: multipoles of W22 (dict from :func:`_window_multipoles`); ``S``: those of conj(W12).
    Returns a list of 6 lists (one per multipole pair of :attr:`kernel_ells`) of 15 complex arrays.
    """
    c = jnp.conj
    W_L0, W_k1L2, W_k2L2, W_k1L4, W_k2L4 = W['L0'], W['k1L2'], W['k2L2'], W['k1L4'], W['k2L4']
    W_k1L2_k2L2, W_k1L2_k2L4, W_k1L4_k2L2, W_k1L4_k2L4 = W['k1L2_k2L2'], W['k1L2_k2L4'], W['k1L4_k2L2'], W['k1L4_k2L4']
    Wc_L0, Wc_k1L2, Wc_k2L2, Wc_k1L4, Wc_k2L4 = c(W_L0), c(W_k1L2), c(W_k2L2), c(W_k1L4), c(W_k2L4)
    Wc_k1L2_k2L2, Wc_k1L2_k2L4, Wc_k1L4_k2L4 = c(W_k1L2_k2L2), c(W_k1L2_k2L4), c(W_k1L4_k2L4)
    Wc_k1L4_k2L2 = Wc_k1L2_k2L4
    W12c_L0, W12_k1L2, W12_k2L2, W12_k1L4, W12_k2L4 = S['L0'], S['k1L2'], S['k2L2'], S['k1L4'], S['k2L4']
    W12_k1L2_k2L2, W12_k1L4_k2L2, W12_k1L4_k2L4 = S['k1L2_k2L2'], S['k1L4_k2L2'], S['k1L4_k2L4']
    if covapt_bugs:
        W12_k1L4 = S['k1L4_covapt']
        W12_k1L4_k2L2 = 2. / 7. * W12_k1L2 + 20. / 77. * W12_k1L4
        W12_k1L4_k2L4 = 1. / 9. * W12c_L0 + 100. / 693. * W12_k1L2 + 162. / 1001. * W12_k1L4

    W_k1L2_Sumk2L22 = 1. / 5. * W_k1L2 + 2. / 7. * W_k1L2_k2L2 + 18. / 35. * W_k1L2_k2L4
    W_k1L2_Sumk2L24 = 2. / 7. * W_k1L2_k2L2 + 20. / 77. * W_k1L2_k2L4
    W_k1L4_Sumk2L22 = 1. / 5. * W_k1L4 + 2. / 7. * W_k1L4_k2L2 + 18. / 35. * W_k1L4_k2L4
    W_k1L4_Sumk2L24 = 2. / 7. * W_k1L4_k2L2 + 20. / 77. * W_k1L4_k2L4
    W_k1L4_Sumk2L44 = 1. / 9. * W_k1L4 + 100. / 693. * W_k1L4_k2L2 + 162. / 1001. * W_k1L4_k2L4

    C00 = [Wc_L0 * W_L0, Wc_L0 * W_k2L2, Wc_L0 * W_k2L4,
           Wc_k1L2 * W_L0, Wc_k1L2 * W_k2L2, Wc_k1L2 * W_k2L4,
           Wc_k1L4 * W_L0, Wc_k1L4 * W_k2L2, Wc_k1L4 * W_k2L4]
    C00 += [2. * W_L0 * W12c_L0, W_k1L2 * W12c_L0, W_k1L4 * W12c_L0, W_k2L2 * W12c_L0, W_k2L4 * W12c_L0, c(W12c_L0) * W12c_L0]

    C22 = [Wc_k2L2 * W_k1L2 + Wc_L0 * W_k1L2_k2L2,
           Wc_k2L2 * W_k1L2_k2L2 + Wc_L0 * W_k1L2_Sumk2L22,
           Wc_k2L2 * W_k1L2_k2L4 + Wc_L0 * W_k1L2_Sumk2L24,
           Wc_k1L2_k2L2 * W_k1L2 + Wc_k1L2 * W_k1L2_k2L2,
           Wc_k1L2_k2L2 * W_k1L2_k2L2 + Wc_k1L2 * W_k1L2_Sumk2L22,
           Wc_k1L2_k2L2 * W_k1L2_k2L4 + Wc_k1L2 * W_k1L2_Sumk2L24,
           Wc_k1L4_k2L2 * W_k1L2 + Wc_k1L4 * W_k1L2_k2L2,
           Wc_k1L4_k2L2 * W_k1L2_k2L2 + Wc_k1L4 * W_k1L2_Sumk2L22,
           Wc_k1L4_k2L2 * W_k1L2_k2L4 + Wc_k1L4 * W_k1L2_Sumk2L24]
    C22 += [W_k1L2 * W12_k2L2 + W_k2L2 * W12_k1L2 + W_k1L2_k2L2 * W12c_L0 + W_L0 * W12_k1L2_k2L2,
            0.5 * ((1. / 5. * W_L0 + 2. / 7. * W_k1L2 + 18. / 35. * W_k1L4) * W12_k2L2 + W_k1L2_k2L2 * W12_k1L2
                   + (1. / 5. * W_k2L2 + 2. / 7. * W_k1L2_k2L2 + 18. / 35. * W_k1L4_k2L2) * W12c_L0 + W_k1L2 * W12_k1L2_k2L2),
            0.5 * ((2. / 7. * W_k1L2 + 20. / 77. * W_k1L4) * W12_k2L2 + W_k1L4_k2L2 * W12_k1L2
                   + (2. / 7. * W_k1L2_k2L2 + 20. / 77. * W_k1L4_k2L2) * W12c_L0 + W_k1L4 * W12_k1L2_k2L2),
            0.5 * (W_k1L2_k2L2 * W12_k2L2 + (1. / 5. * W_L0 + 2. / 7. * W_k2L2 + 18. / 35. * W_k2L4) * W12_k1L2
                   + (1. / 5. * W_k1L2 + 2. / 7. * W_k1L2_k2L2 + 18. / 35. * W_k1L2_k2L4) * W12c_L0 + W_k2L2 * W12_k1L2_k2L2),
            0.5 * (W_k1L2_k2L4 * W12_k2L2 + (2. / 7. * W_k2L2 + 20. / 77. * W_k2L4) * W12_k1L2
                   + W_k2L4 * W12_k1L2_k2L2 + (2. / 7. * W_k1L2_k2L2 + 20. / 77. * W_k1L2_k2L4) * W12c_L0),
            c(W12_k1L2_k2L2) * W12c_L0 + c(W12_k1L2) * W12_k2L2]

    C44 = [Wc_k2L4 * W_k1L4 + Wc_L0 * W_k1L4_k2L4,
           Wc_k2L4 * W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L24,
           Wc_k2L4 * W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L44,
           Wc_k1L2_k2L4 * W_k1L4 + Wc_k1L2 * W_k1L4_k2L4,
           Wc_k1L2_k2L4 * W_k1L4_k2L2 + Wc_k1L2 * W_k1L4_Sumk2L24,
           Wc_k1L2_k2L4 * W_k1L4_k2L4 + Wc_k1L2 * W_k1L4_Sumk2L44,
           Wc_k1L4_k2L4 * W_k1L4 + Wc_k1L4 * W_k1L4_k2L4,
           Wc_k1L4_k2L4 * W_k1L4_k2L2 + Wc_k1L4 * W_k1L4_Sumk2L24,
           Wc_k1L4_k2L4 * W_k1L4_k2L4 + Wc_k1L4 * W_k1L4_Sumk2L44]
    C44 += [W_k1L4 * W12_k2L4 + W_k2L4 * W12_k1L4 + W_k1L4_k2L4 * W12c_L0 + W_L0 * W12_k1L4_k2L4,
            0.5 * ((2. / 7. * W_k1L2 + 20. / 77. * W_k1L4) * W12_k2L4 + W_k1L2_k2L4 * W12_k1L4
                   + (2. / 7. * W_k1L2_k2L4 + 20. / 77. * W_k1L4_k2L4) * W12c_L0 + W_k1L2 * W12_k1L4_k2L4),
            0.5 * ((1. / 9. * W_L0 + 100. / 693. * W_k1L2 + 162. / 1001. * W_k1L4) * W12_k2L4 + W_k1L4_k2L4 * W12_k1L4
                   + (1. / 9. * W_k2L4 + 100. / 693. * W_k1L2_k2L4 + 162. / 1001. * W_k1L4_k2L4) * W12c_L0 + W_k1L4 * W12_k1L4_k2L4),
            0.5 * (W_k1L4_k2L2 * W12_k2L4 + (2. / 7. * W_k2L2 + 20. / 77. * W_k2L4) * W12_k1L4
                   + W_k2L2 * W12_k1L4_k2L4 + (2. / 7. * W_k1L4_k2L2 + 20. / 77. * W_k1L4_k2L4) * W12c_L0),
            0.5 * (W_k1L4_k2L4 * W12_k2L4 + (1. / 9. * W_L0 + 100. / 693. * W_k2L2 + 162. / 1001. * W_k2L4) * W12_k1L4
                   + W_k2L4 * W12_k1L4_k2L4 + (1. / 9. * W_k1L4 + 100. / 693. * W_k1L4_k2L2 + 162. / 1001. * W_k1L4_k2L4) * W12c_L0),
            c(W12_k1L4_k2L4) * W12c_L0 + c(W12_k1L4) * W12_k2L4]

    C20 = [Wc_L0 * W_k1L2, Wc_L0 * W_k1L2_k2L2, Wc_L0 * W_k1L2_k2L4,
           Wc_k1L2 * W_k1L2, Wc_k1L2 * W_k1L2_k2L2, Wc_k1L2 * W_k1L2_k2L4,
           Wc_k1L4 * W_k1L2, Wc_k1L4 * W_k1L2_k2L2, Wc_k1L4 * W_k1L2_k2L4]
    C20 += [W_k1L2 * W12c_L0 + W_L0 * W12_k1L2,
            0.5 * ((1. / 5. * W_L0 + 2. / 7. * W_k1L2 + 18. / 35. * W_k1L4) * W12c_L0 + W_k1L2 * W12_k1L2),
            0.5 * ((2. / 7. * W_k1L2 + 20. / 77. * W_k1L4) * W12c_L0 + W_k1L4 * W12_k1L2),
            0.5 * (W_k1L2_k2L2 * W12c_L0 + W_k2L2 * W12_k1L2),
            0.5 * (W_k1L2_k2L4 * W12c_L0 + W_k2L4 * W12_k1L2),
            c(W12_k1L2) * W12c_L0]

    C40 = [Wc_L0 * W_k1L4, Wc_L0 * W_k1L4_k2L2, Wc_L0 * W_k1L4_k2L4,
           Wc_k1L2 * W_k1L4, Wc_k1L2 * W_k1L4_k2L2, Wc_k1L2 * W_k1L4_k2L4,
           Wc_k1L4 * W_k1L4, Wc_k1L4 * W_k1L4_k2L2, Wc_k1L4 * W_k1L4_k2L4]
    C40 += [W_k1L4 * W12c_L0 + W_L0 * W12_k1L4,
            0.5 * ((2. / 7. * W_k1L2 + 20. / 77. * W_k1L4) * W12c_L0 + W_k1L2 * W12_k1L4),
            0.5 * ((1. / 9. * W_L0 + 100. / 693. * W_k1L2 + 162. / 1001. * W_k1L4) * W12c_L0 + W_k1L4 * W12_k1L4),
            0.5 * (W_k1L4_k2L2 * W12c_L0 + W_k2L2 * W12_k1L4),
            0.5 * (W_k1L4_k2L4 * W12c_L0 + W_k2L4 * W12_k1L4),
            c(W12_k1L4) * W12c_L0]

    C42 = [Wc_k2L2 * W_k1L4 + Wc_L0 * W_k1L4_k2L2,
           Wc_k2L2 * W_k1L4_k2L2 + Wc_L0 * W_k1L4_Sumk2L22,
           Wc_k2L2 * W_k1L4_k2L4 + Wc_L0 * W_k1L4_Sumk2L24,
           Wc_k1L2_k2L2 * W_k1L4 + Wc_k1L2 * W_k1L4_k2L2,
           Wc_k1L2_k2L2 * W_k1L4_k2L2 + Wc_k1L2 * W_k1L4_Sumk2L22,
           Wc_k1L2_k2L2 * W_k1L4_k2L4 + Wc_k1L2 * W_k1L4_Sumk2L24,
           Wc_k1L4_k2L2 * W_k1L4 + Wc_k1L4 * W_k1L4_k2L2,
           Wc_k1L4_k2L2 * W_k1L4_k2L2 + Wc_k1L4 * W_k1L4_Sumk2L22,
           Wc_k1L4_k2L2 * W_k1L4_k2L4 + Wc_k1L4 * W_k1L4_Sumk2L24]
    C42 += [W_k1L4 * W12_k2L2 + W_k2L2 * W12_k1L4 + W_k1L4_k2L2 * W12c_L0 + W_L0 * W12_k1L4_k2L2,
            0.5 * ((2. / 7. * W_k1L2 + 20. / 77. * W_k1L4) * W12_k2L2 + W_k1L2_k2L2 * W12_k1L4
                   + (2. / 7. * W_k1L2_k2L2 + 20. / 77. * W_k1L4_k2L2) * W12c_L0 + W_k1L2 * W12_k1L4_k2L2),
            0.5 * ((1. / 9. * W_L0 + 100. / 693. * W_k1L2 + 162. / 1001. * W_k1L4) * W12_k2L2 + W_k1L4_k2L2 * W12_k1L4
                   + (1. / 9. * W_k2L2 + 100. / 693. * W_k1L2_k2L2 + 162. / 1001. * W_k1L4_k2L2) * W12c_L0 + W_k1L4 * W12_k1L4_k2L2),
            0.5 * (W_k1L4_k2L2 * W12_k2L2 + (1. / 5. * W_L0 + 2. / 7. * W_k2L2 + 18. / 35. * W_k2L4) * W12_k1L4
                   + W_k2L2 * W12_k1L4_k2L2 + (1. / 5. * W_k1L4 + 2. / 7. * W_k1L4_k2L2 + 18. / 35. * W_k1L4_k2L4) * W12c_L0),
            0.5 * (W_k1L4_k2L4 * W12_k2L2 + (2. / 7. * W_k2L2 + 20. / 77. * W_k2L4) * W12_k1L4
                   + W_k2L4 * W12_k1L4_k2L2 + (2. / 7. * W_k1L4_k2L2 + 20. / 77. * W_k1L4_k2L4) * W12c_L0),
            c(W12_k1L4_k2L2) * W12c_L0 + c(W12_k1L4) * W12_k2L2]
    return [C00, C22, C44, C20, C40, C42]


def _covapt_W12_k1L4(S4c, k1h, W12_k1L2, W12c_L0):
    """CovaPT's ``W12_k1L4``, which omits the yyyz, xzzz and yzzz cubic terms (kept for comparison only)."""
    (xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz) = S4c
    x, y, z = k1h
    return 35. / 8. * (xxxx * x**4 + yyyy * y**4 + zzzz * z**4 + 4. * xxxy * x**3 * y + 4. * xxxz * x**3 * z + 4. * xyyy * y**3 * x
                       + 6. * xxyy * x**2 * y**2 + 6. * xxzz * x**2 * z**2 + 6. * yyzz * y**2 * z**2
                       + 12. * xxyz * x**2 * y * z + 12. * xyyz * y**2 * x * z + 12. * xyzz * z**2 * x * y) \
        - 5. / 2. * W12_k1L2 - 7. / 8. * W12c_L0


def _kernel_one_mode(W22, W12c, ikgrid, ik1, ik1r, k1_bin, kfun, dk, kmin_index, nk, delta_k_max, covapt_bugs):
    """Sum of the 90 kernel integrands over the k2 = k1 - q modes of the FFT grid, binned by delta = bin(k2) - bin(k1)."""
    k1h = jnp.where(ik1r > 1e-10, ik1 / jnp.where(ik1r > 1e-10, ik1r, 1.), 0.)
    k2 = ik1[:, None, None, None] - ikgrid
    k2r = jnp.sqrt(jnp.sum(k2**2, axis=0))
    k2_bin = jnp.floor(k2r * kfun / dk).astype(int) - kmin_index
    delta = k2_bin - k1_bin
    valid = (k2_bin >= 0) & (k2_bin < nk) & (jnp.abs(delta) <= delta_k_max)
    nseg = 2 * delta_k_max + 1
    iseg = jnp.where(valid, delta + delta_k_max, nseg)
    k2rs = jnp.where(k2r > 1e-10, k2r, jnp.inf)
    k2h = k2 / k2rs
    W = _window_multipoles(W22[0], W22[1], W22[2], k1h, k2h, *W22[3:])
    S = _window_multipoles(W12c[0], W12c[1], W12c[2], k1h, k2h, *W12c[3:])
    if covapt_bugs:
        S['k1L4_covapt'] = _covapt_W12_k1L4(W12c[2], k1h, S['k1L2'], S['L0'])
    terms = _kernel_terms(W, S, covapt_bugs=covapt_bugs)
    data = jnp.stack([t.real.ravel() for C in terms for t in C], axis=-1)  # (n^3, 90)
    onehot = (iseg.ravel()[:, None] == jnp.arange(nseg)[None, :]).astype(data.dtype)  # (n^3, nseg)
    out = onehot.T @ data  # (nseg, 90)
    return out.reshape(nseg, 6, 15).transpose(0, 2, 1)  # (nseg, 15, 6)


def compute_window_kernels(window, kedges, kmodes_sampled=2500, kmodes=None, nmodes=None, k_shell_approx=0.1,
                           seed=42, batch_size=8, delta_k_max=None, covapt_bugs=False, exact_high_order=True, dense=True,
                           return_error=False, progress=True):
    r"""
    Window kernels of the Gaussian covariance (CovaPT ``WinFun``, ``thecov.SurveyGeometry.compute_window_kernels``).

    Parameters
    ----------
    window : SurveyWindow
    kedges : array
        Linear k-bin edges of the covariance; ``kmin`` should be a multiple of ``dk``.
    kmodes_sampled : int, default=2500
        Number of :math:`k_1` modes sampled per bin (Monte-Carlo average over the :math:`k_1` shell).
    kmodes, nmodes : list, array, optional
        Sampled modes and mode counts per bin (from :func:`sample_kmodes`); by default drawn here with ``seed``.
    k_shell_approx : float, default=0.1
        Above this wavenumber the mode count of the box is the continuum estimate and modes are drawn from the
        continuous shell (as ``thecov``); ``None`` (or ``inf``) for exact lattice counts and sampling everywhere.
    batch_size : int, default=8
        Number of sampled modes processed at once (memory ~ ``batch_size`` x 100 x mesh size x 16 bytes).
    delta_k_max : int, optional
        Number of k-bins on each side of the diagonal; default ``window.delta_k_max = nmesh // 2 - 1``.
    covapt_bugs : bool, default=False
        Reproduce CovaPT's / thecov's ``W12_k1L4`` (missing three cubic moments), for comparisons.
    exact_high_order : bool, default=True
        Paint the 6th and 8th order moments of the windows (73 more per window) and use the exact products
        L2(k1) L4(k2), L4(k1) L2(k2), L4(k1) L4(k2). ``False`` uses CovaPT's / thecov's approximation (k2 -> k1, L6 and
        L8 dropped), which only affects the kernels of the (2,2), (4,4) and (2,4) blocks (and (0,4) through the sums)
        but makes the three-multipole covariance indefinite (VALIDATION.md, section 6).
    dense : bool, default=True
        Return kernels as ``(nk, nk, 15, 6)`` (``[i, j]`` = bins of :math:`k_1, k_2`); else in ``thecov`` layout
        ``(nk, 2 * delta_k_max + 1, 15, 6)`` with ``j - i + delta_k_max`` on axis 1.

    Returns
    -------
    kernels : array
        Normalized as CovaPT: mean over sampled :math:`k_1`, divided by :math:`N_\mathrm{modes}(k_2)` and multiplied by
        :math:`(2 \ell_1 + 1)(2 \ell_2 + 1)` (times 2 if one multipole is 0). Multiply by :math:`1 / I_{22}^2` and the
        power spectra to get the covariance (:func:`gaussian_covariance_survey`).
    error : array, optional
        Relative Monte-Carlo error of the kernels (std over batches / sqrt(nbatches) / mean), if ``return_error``.
    """
    kedges, kmin, dk = _check_linear_edges(kedges)
    nk = len(kedges) - 1
    kfun = window.kfun
    if delta_k_max is None:
        delta_k_max = window.delta_k_max
    delta_k_max = int(delta_k_max)
    nseg = 2 * delta_k_max + 1
    kmin_index = int(np.round(kmin / dk))
    if kmodes is None:
        k_shell = np.inf if k_shell_approx is None else k_shell_approx
        kmodes, nmodes = sample_kmodes(kedges, window.boxsize, max_modes=kmodes_sampled, k_shell_approx=k_shell, seed=seed)
    elif nmodes is None:
        nmodes = count_modes(kedges, window.boxsize, exact_below=k_shell_approx)
    nmodes = np.asarray(nmodes, dtype=float)
    assert len(kmodes) == nk and len(nmodes) == nk

    orders = (0, 2, 4, 6, 8) if exact_high_order else (0, 2, 4)
    W22 = window.compute_window_ffts('22', orders=orders)
    W12 = window.compute_window_ffts('12', orders=orders)
    W12c = tuple(jnp.conj(w) if w is not None else None for w in W12)  # CovaPT uses conj(W12)
    ikgrid = jnp.array(window.ikgrid(), dtype=float)

    def batch(ik, ikr, k1_bin, weight):
        f = jax.vmap(lambda ik, ikr: _kernel_one_mode(W22, W12c, ikgrid, ik, ikr, k1_bin, kfun, dk, kmin_index, nk, delta_k_max, covapt_bugs))
        return jnp.sum(f(ik, ikr) * weight[:, None, None, None], axis=0)

    batch = jax.jit(batch)
    ell_factor = np.array([(2 * l1 + 1) * (2 * l2 + 1) * (2 if 0 in (l1, l2) else 1) for l1, l2 in kernel_ells])
    kernels = np.zeros((nk, nseg, 15, 6))
    errors = np.zeros((nk, nseg, 15, 6))
    iterator = range(nk)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='window kernels')
        except ImportError:
            pass
    for i in iterator:
        km = np.asarray(kmodes[i], dtype=float)
        nm = len(km)
        results = []
        for start in range(0, nm, batch_size):
            chunk = km[start:start + batch_size]
            weight = np.ones(len(chunk))
            if len(chunk) < batch_size:  # pad to a fixed shape
                pad = batch_size - len(chunk)
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], pad, axis=0)])
                weight = np.concatenate([weight, np.zeros(pad)])
            results.append(np.asarray(batch(jnp.array(chunk[:, :3]), jnp.array(chunk[:, 3]), i, jnp.array(weight))))
        results = np.array(results)
        total = results.sum(axis=0) / nm
        if return_error and len(results) > 1:
            per_mode = results / np.maximum(np.array([min(batch_size, nm - s) for s in range(0, nm, batch_size)]), 1)[:, None, None, None]
            std = per_mode.std(axis=0) / np.sqrt(len(results))
            mean = per_mode.mean(axis=0)
            errors[i] = np.where(std == 0, 0., std / np.where(mean == 0, 1., mean))
        for s in range(nseg):
            j = i + s - delta_k_max
            if 0 <= j < nk:
                total[s] /= nmodes[j]
            else:
                total[s] = 0.
        kernels[i] = total * ell_factor
    if dense:
        kernels = kernels_to_dense(kernels)
        errors = kernels_to_dense(errors)
    if return_error:
        return kernels, errors
    return kernels


def kernels_to_dense(kernels):
    """Convert kernels from the ``thecov`` layout ``(nk, 2 dkmax + 1, ...)`` to ``(nk, nk, ...)``."""
    nk, nseg = kernels.shape[:2]
    dkmax = (nseg - 1) // 2
    dense = np.zeros((nk, nk) + kernels.shape[2:])
    for i in range(nk):
        for s in range(nseg):
            j = i + s - dkmax
            if 0 <= j < nk:
                dense[i, j] = kernels[i, s]
    return dense


def kernels_to_thecov(kernels, delta_k_max):
    """Convert dense kernels ``(nk, nk, ...)`` to the ``thecov`` layout ``(nk, 2 delta_k_max + 1, ...)``."""
    nk = kernels.shape[0]
    nseg = 2 * delta_k_max + 1
    out = np.zeros((nk, nseg) + kernels.shape[2:])
    for i in range(nk):
        for s in range(nseg):
            j = i + s - delta_k_max
            if 0 <= j < nk:
                out[i, s] = kernels[i, j]
    return out


def gaussian_covariance_survey(kernels, pk, alpha=0., I22=1., pk_renorm=1., ells=(0, 2, 4), return_terms=False):
    r"""
    Gaussian covariance of :math:`P_{0, 2, 4}` from the window kernels (CovaPT ``Cij``).

    Parameters
    ----------
    kernels : array (nk, nk, 15, 6)
        Dense window kernels from :func:`compute_window_kernels`.
    pk : array (3, nk)
        :math:`P_0, P_2, P_4` (in the estimator's normalization), :math:`P_0` **without** shot noise.
    alpha : float
        :math:`N_\mathrm{data} / N_\mathrm{randoms}` of the estimator: the shot-noise terms carry :math:`(1 + \alpha)`
        and :math:`(1 + \alpha)^2`; the effective shot noise is :math:`(1 + \alpha) I_{12} / I_{22}`.
    I22 : float
        Normalization :math:`I_{22} = \int \bar n^2 w^2` of the windows (the kernels are not divided by it).
    pk_renorm : float
        Ratio of the estimator's normalization to :math:`I_{22}` (``thecov``'s ``pk_renorm``): the kernels are
        multiplied by :math:`(\mathrm{pk\_renorm} / I_{22})^2` and the spectra divided by ``pk_renorm``.

    Returns
    -------
    blocks : array (3, 3, nk, nk)
        Blocks ``[il1, il2]`` for ``ells``; ``blocks_to_matrix`` stacks them. If ``return_terms``, a dict with
        the cosmic-variance, mixed and shot-noise contributions separately.
    """
    kernels = np.asarray(kernels)
    nk = kernels.shape[0]
    P0, P2, P4 = [np.asarray(p, dtype=float) / pk_renorm for p in pk]
    K = [kernels[:, :, t, :] for t in range(15)]  # each (nk, nk, 6)
    o = lambda a, b: np.outer(a, b)[..., None]
    cv = (K[0] * o(P0, P0) + K[1] * o(P0, P2) + K[2] * o(P0, P4)
          + K[3] * o(P2, P0) + K[4] * o(P2, P2) + K[5] * o(P2, P4)
          + K[6] * o(P4, P0) + K[7] * o(P4, P2) + K[8] * o(P4, P4))
    mixed = (K[9] * (P0[:, None] + P0[None, :])[..., None] / 2. + K[10] * P2[:, None, None] + K[11] * P4[:, None, None]
             + K[12] * P2[None, :, None] + K[13] * P4[None, :, None])
    shot = K[14]
    norm = (pk_renorm / I22)**2
    terms = {'cosmic_variance': cv * norm, 'mixed': (1 + alpha) * mixed * norm, 'shotnoise': (1 + alpha)**2 * shot * norm}

    def to_blocks(cov6):
        blocks = np.zeros((3, 3, nk, nk))
        for m, (l1, l2) in enumerate(kernel_ells):
            i1, i2 = l1 // 2, l2 // 2
            blocks[i1, i2] = cov6[..., m]
            if i1 != i2:
                blocks[i2, i1] = cov6[..., m].T
        return blocks

    if return_terms:
        return {name: to_blocks(value) for name, value in terms.items()}
    return to_blocks(sum(terms.values()))


def gaussian_covariance_box(pk, nmodes, shotnoise=0.):
    r"""
    Gaussian covariance of :math:`P_{0, 2, 4}` in a periodic box, :math:`2 / N_\mathrm{modes}` times the products of
    multipoles (``thecov.GaussianCovariance._compute_covariance_box``); ``pk[0]`` is :math:`P_0` without shot noise,
    ``shotnoise`` is added to it.

    Returns
    -------
    blocks : array (3, 3, nk, nk), diagonal.
    """
    P0, P2, P4 = [np.asarray(p, dtype=float) for p in pk]
    P0 = P0 + shotnoise
    cov = {}
    cov[0, 0] = P0**2 + 1 / 5 * P2**2 + 1 / 9 * P4**2
    cov[0, 2] = 2 * P0 * P2 + 2 / 7 * P2**2 + 4 / 7 * P2 * P4 + 100 / 693 * P4**2
    cov[0, 4] = 2 * P0 * P4 + 18 / 35 * P2**2 + 40 / 77 * P2 * P4 + 162 / 1001 * P4**2
    cov[2, 2] = 5 * P0**2 + 20 / 7 * P0 * P2 + 20 / 7 * P0 * P4 + 15 / 7 * P2**2 + 120 / 77 * P2 * P4 + 8945 / 9009 * P4**2
    cov[2, 4] = 36 / 7 * P0 * P2 + 200 / 77 * P0 * P4 + 108 / 77 * P2**2 + 3578 / 1001 * P2 * P4 + 900 / 1001 * P4**2
    cov[4, 4] = 9 * P0**2 + 360 / 77 * P0 * P2 + 2916 / 1001 * P0 * P4 + 16101 / 5005 * P2**2 + 3240 / 1001 * P2 * P4 + 42849 / 17017 * P4**2
    nk = len(P0)
    blocks = np.zeros((3, 3, nk, nk))
    for (l1, l2), value in cov.items():
        blocks[l1 // 2, l2 // 2] = np.diag(2. / np.asarray(nmodes, dtype=float) * value)
        blocks[l2 // 2, l1 // 2] = blocks[l1 // 2, l2 // 2]
    return blocks


def blocks_to_matrix(blocks):
    """``(nells, nells, nk, nk)`` -> ``(nells * nk, nells * nk)``."""
    blocks = np.asarray(blocks)
    return np.block([[blocks[i, j] for j in range(blocks.shape[1])] for i in range(blocks.shape[0])])


def matrix_to_blocks(matrix, nells=3):
    """``(nells * nk, nells * nk)`` -> ``(nells, nells, nk, nk)``."""
    matrix = np.asarray(matrix)
    nk = matrix.shape[0] // nells
    return np.array([[matrix[i * nk:(i + 1) * nk, j * nk:(j + 1) * nk] for j in range(nells)] for i in range(nells)])


def to_covariance_matrix(blocks, k, kedges=None, ells=(0, 2, 4), nmodes=None):
    """Wrap covariance blocks into an :class:`lsstypes.CovarianceMatrix` with a ``Mesh2SpectrumPoles`` observable."""
    from .types import CovarianceMatrix, Mesh2SpectrumPole, Mesh2SpectrumPoles
    k = np.asarray(k)
    if kedges is None:
        dk = np.diff(k).mean()
        kedges = np.concatenate([k - dk / 2, [k[-1] + dk / 2]])
    kedges = np.asarray(kedges)
    edges = np.column_stack([kedges[:-1], kedges[1:]]) if kedges.ndim == 1 else kedges
    poles = []
    for ell in ells:
        kw = {}
        if nmodes is not None: kw['nmodes'] = np.asarray(nmodes)
        poles.append(Mesh2SpectrumPole(k=k, k_edges=edges, num_raw=np.zeros_like(k), num_shotnoise=np.zeros_like(k), ell=ell, **kw))
    return CovarianceMatrix(observable=Mesh2SpectrumPoles(poles), value=blocks_to_matrix(blocks))


# ----------------------------------------------------------------------------------------------------------------------
# Regular trispectrum T0 (Kobayashi 2023, Wadekar & Scoccimarro 2020 Eq. 66)
# ----------------------------------------------------------------------------------------------------------------------

def pbs_bias_parameters(b1, b2=None, bG2=None, b3=None, bG3=None, bdG2=None, bGamma3=None):
    r"""
    Bias parameters in the basis of Kobayashi (2023, Eq. A1) from :math:`b_1`, filling the missing ones with the
    local-Lagrangian / peak-background-split relations used by CovaPT and ``thecov`` (``set_params``):
    :math:`\gamma_2 = -2/7 (b_1 - 1)`, :math:`\gamma_3 = 11/63 (b_1 - 1)`, :math:`b_2` from the fit of
    Lazeyras et al. (2016) rescaled to the :math:`\gamma_2` basis, :math:`\gamma_2^\times = -2/7 b_2`,
    :math:`\gamma_{21} = -22/147 (b_1 - 1)`, :math:`b_3` from the fit of Lazeyras et al. (2016), and
    :math:`b_{\Gamma_3} = -4/7 \gamma_{21}` (Eq. 44 of arXiv:1812.03208, as ``thecov``).
    """
    g2 = -2 / 7 * (b1 - 1) if bG2 is None else bG2
    g3 = 11 / 63 * (b1 - 1) if bG3 is None else bG3
    if b2 is None:
        b2 = 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3 + 4 / 3 * g2
    g2x = -2 / 7 * b2 if bdG2 is None else bdG2
    g21 = -22 / 147 * (b1 - 1)
    if b3 is None:
        b3 = -1.028 + 7.646 * b1 - 6.227 * b1**2 + 0.912 * b1**3 + 4 * g2x - 4 / 3 * g3 - 8 / 3 * g21 - 32 / 21 * g2
    if bGamma3 is None:
        bGamma3 = -4 / 7 * g21
    return dict(b1=float(b1), b2=float(b2), bG2=float(g2), b3=float(b3), bG3=float(g3), bdG2=float(g2x), bGamma3=float(bGamma3))


def _vdot(a, b):
    return jnp.sum(a * b, axis=-1)


def _vnorm(a):
    return jnp.sqrt(_vdot(a, a))


def _safe_div(num, den, tiny=1e-14):
    ok = jnp.abs(den) > tiny
    return jnp.where(ok, num / jnp.where(ok, den, 1.), 0.)


class _PTKernels(object):
    """Redshift-space PT kernels of Kobayashi (2023), Appendix A, for vector arguments ``(..., 3)`` and line of sight ``n`` ``(..., 3)``."""

    def __init__(self, f, bias):
        self.f = f
        self.b = bias

    def mu(self, k, n):
        return _safe_div(_vdot(k, n), _vnorm(k))

    def F2(self, k1, k2):
        n1, n2 = _vnorm(k1), _vnorm(k2)
        x = _safe_div(_vdot(k1, k2), n1 * n2)
        return 5. / 7. + x / 2. * (_safe_div(n1, n2) + _safe_div(n2, n1)) + 2. / 7. * x**2

    def G2(self, k1, k2):
        n1, n2 = _vnorm(k1), _vnorm(k2)
        x = _safe_div(_vdot(k1, k2), n1 * n2)
        return 3. / 7. + x / 2. * (_safe_div(n1, n2) + _safe_div(n2, n1)) + 4. / 7. * x**2

    def KG2(self, k1, k2):
        n1, n2 = _vnorm(k1), _vnorm(k2)
        x = _safe_div(_vdot(k1, k2), n1 * n2)
        # 0 (instead of -1) when one vector vanishes: KG2 always multiplies F2 or G2 of vectors summing to k2 in
        # that case, which vanish
        return jnp.where((n1 > 1e-14) & (n2 > 1e-14), x**2 - 1., 0.)

    def KG3(self, k1, k2, k3):
        n1, n2, n3 = _vnorm(k1), _vnorm(k2), _vnorm(k3)
        x12, x23, x31 = _safe_div(_vdot(k1, k2), n1 * n2), _safe_div(_vdot(k2, k3), n2 * n3), _safe_div(_vdot(k3, k1), n3 * n1)
        return 2 * x12 * x23 * x31 - (x12**2 + x23**2 + x31**2) + 1.

    @staticmethod
    def _alpha(ka, kb):
        r"""Vertex :math:`\alpha(k_a, k_b) = 1 + k_a \cdot k_b / k_a^2` (0 if a vector vanishes: it then multiplies a vanishing kernel)."""
        na, nb = _vnorm(ka), _vnorm(kb)
        ok = (na > 1e-14) & (nb > 1e-14)
        return jnp.where(ok, 1. + _safe_div(_vdot(ka, kb), na**2), 0.)

    @staticmethod
    def _beta(ka, kb):
        r"""Vertex :math:`\beta(k_a, k_b) = (k_a \cdot k_b) |k_a + k_b|^2 / (2 k_a^2 k_b^2)`."""
        na, nb = _vnorm(ka), _vnorm(kb)
        return _safe_div(_vdot(ka, kb) * _vdot(ka + kb, ka + kb), 2. * na**2 * nb**2)

    def _F3_unsym(self, k1, k2, k3, velocity=False):
        r"""
        Unsymmetrized third-order kernel from the recursion of Bernardeau et al. (2002, Eqs. 43-44):
        :math:`F_3 = \frac{1}{18}[7 \alpha(k_1, k_{23}) F_2(k_2, k_3) + 2 \beta(k_1, k_{23}) G_2(k_2, k_3)
        + G_2(k_1, k_2)(7 \alpha(k_{12}, k_3) + 2 \beta(k_{12}, k_3))]`, and :math:`G_3` with (3, 6) instead of (7, 2).
        """
        k12, k23 = k1 + k2, k2 + k3
        a1, b1 = self._alpha(k1, k23), self._beta(k1, k23)
        a2, b2 = self._alpha(k12, k3), self._beta(k12, k3)
        F2_23, G2_23, G2_12 = self.F2(k2, k3), self.G2(k2, k3), self.G2(k1, k2)
        cf, cg = (3., 6.) if velocity else (7., 2.)
        return (cf * a1 * F2_23 + cg * b1 * G2_23 + G2_12 * (cf * a2 + cg * b2)) / 18.

    def F3(self, k1, k2, k3, velocity=False):
        toret = 0.
        for a, b, c in itertools.permutations((k1, k2, k3)):
            toret = toret + self._F3_unsym(a, b, c, velocity=velocity)
        return toret / 6.

    def G3(self, k1, k2, k3):
        return self.F3(k1, k2, k3, velocity=True)

    def Z1(self, k, n):
        return self.b['b1'] + self.f * self.mu(k, n)**2

    def Z2(self, k1, k2, n):
        b, f = self.b, self.f
        n1, n2 = _vnorm(k1), _vnorm(k2)
        mu1, mu2 = self.mu(k1, n), self.mu(k2, n)
        k12 = k1 + k2
        mu12k12 = _vdot(k12, n)
        mu12 = _safe_div(mu12k12, _vnorm(k12))
        return (b['b1'] * self.F2(k1, k2) + b['b2'] / 2. + b['bG2'] * self.KG2(k1, k2) + f * mu12**2 * self.G2(k1, k2)
                + f * mu12k12 / 2. * b['b1'] * (_safe_div(mu1, n1) + _safe_div(mu2, n2))
                + (f * mu12k12)**2 / 2. * _safe_div(mu1, n1) * _safe_div(mu2, n2))

    def _Z3_unsym(self, k1, k2, k3, n):
        """Eq. (A8) of Kobayashi (2023): first argument distinguished."""
        b, f = self.b, self.f
        n1, n2, n3 = _vnorm(k1), _vnorm(k2), _vnorm(k3)
        mu1, mu2, mu3 = self.mu(k1, n), self.mu(k2, n), self.mu(k3, n)
        k23 = k2 + k3
        k123 = k1 + k23
        n23 = _vnorm(k23)
        mu23 = self.mu(k23, n)
        muk = _vdot(k123, n)  # mu_123 k_123
        mu123 = _safe_div(muk, _vnorm(k123))
        F2_23, G2_23 = self.F2(k2, k3), self.G2(k2, k3)
        KG2_1_23 = self.KG2(k1, k23)
        toret = (b['b1'] * self.F3(k1, k2, k3) + b['b2'] * F2_23 + 2 * b['bG2'] * KG2_1_23 * F2_23 + b['b3'] / 6.
                 + b['bdG2'] * self.KG2(k2, k3) + b['bG3'] * self.KG3(k1, k2, k3)
                 + 2 * b['bGamma3'] * KG2_1_23 * (F2_23 - G2_23)
                 + f * mu123**2 * self.G3(k1, k2, k3)
                 + f * muk * b['b1'] * _safe_div(mu23, n23) * G2_23
                 + f * muk * _safe_div(mu1, n1) * (b['b1'] * F2_23 + b['b2'] / 2. + b['bG2'] * self.KG2(k2, k3))
                 + (f * muk)**2 * _safe_div(mu1, n1) * _safe_div(mu23, n23) * G2_23
                 + (f * muk)**2 / 2. * b['b1'] * _safe_div(mu1, n1) * _safe_div(mu2, n2)
                 + (f * muk)**3 / 6. * _safe_div(mu1, n1) * _safe_div(mu2, n2) * _safe_div(mu3, n3))
        return toret

    def Z3(self, k1, k2, k3, n):
        """Symmetrized third-order kernel."""
        toret = 0.
        for a, b, c in itertools.permutations((k1, k2, k3)):
            toret = toret + self._Z3_unsym(a, b, c, n)
        return toret / 6.


def _log_interpolator(k, pk):
    logk, logpk = jnp.log(jnp.asarray(k)), jnp.log(jnp.asarray(pk))

    def func(x):
        return jnp.exp(jnp.interp(jnp.log(x), logk, logpk, left='extrapolate', right='extrapolate'))

    return func


def _tabulate_pk(pk_lin, k=None, kmin=1e-6, kmax=1e3, n=1024):
    """Tabulate a callable / (k, pk) pair on a log grid with power-law extrapolation."""
    if callable(pk_lin):
        kk = np.geomspace(kmin, kmax, n)
        pp = np.asarray(pk_lin(kk), dtype=float)
        ok = np.isfinite(pp) & (pp > 0)
        kk, pp = kk[ok], pp[ok]
    else:
        kk, pp = np.asarray(k, dtype=float), np.asarray(pk_lin, dtype=float)
    return kk, pp


def _los_grid(ntheta, nphi):
    """Product rule on the sphere, exact for polynomials of degree < 2 ntheta in cos(theta) and trigonometric degree < nphi."""
    ct, wt = np.polynomial.legendre.leggauss(ntheta)
    phi = 2 * np.pi * (np.arange(nphi) + 0.5) / nphi
    st = np.sqrt(1 - ct**2)
    n = np.array([np.outer(st, np.cos(phi)).ravel(), np.outer(st, np.sin(phi)).ravel(), np.repeat(ct, nphi)]).T
    w = np.repeat(wt, nphi) / (2. * nphi)  # sum(w) = 1: average over the sphere
    return n, w


def compute_trispectrum_covariance(k, pk_lin, f, bias, volume, ndens=None, ndens2=None, ells=(0, 2, 4), include_shotnoise=True,
                                   nq=128, ntheta=12, nphi=24, qmin_factor=1e-4, kedges=None, return_terms=False):
    r"""
    Regular-trispectrum covariance :math:`C^{T_0}_{\ell_1 \ell_2}(k_1, k_2)` of Kobayashi (2023) Eq. (7), with the
    shot-noise part of Appendix C (Eq. C4), by direct numerical angular integration.

    Parameters
    ----------
    k : array (nk,)
        Bin centres (the thin-shell approximation :math:`|k_1| = k_1` is used, as ``powercovfft`` / ``thecov``).
    pk_lin : callable or array
        Linear **matter** power spectrum (bias is in ``bias``); a callable ``pk_lin(k)``, or tabulated values on
        ``kedges`` (then a k-array).
    f : float
        Growth rate.
    bias : dict
        ``{'b1', 'b2', 'bG2', 'b3', 'bG3', 'bdG2', 'bGamma3'}`` (:func:`pbs_bias_parameters`).
    volume : float
        :math:`V` (box) or :math:`I_{22}^2 / I_{44}` (survey).
    ndens, ndens2 : float, optional
        :math:`\bar n` and :math:`\bar n^2` (box, ``ndens2`` defaults to ``ndens**2``); :math:`I_{44} / I_{34}` and
        :math:`I_{44} / I_{24}` (survey). Required if ``include_shotnoise``.
    nq : int
        Number of Gauss-Legendre nodes in :math:`\ln q`, :math:`q = |k_1 + k_2|`, for the integral over
        :math:`\hat k_1 \cdot \hat k_2`.
    ntheta, nphi : int
        Line-of-sight product rule (exact for the polynomial integrand if ``2 ntheta > 21`` and ``nphi > 21``).
    qmin_factor : float
        Lower cut :math:`q_\mathrm{min} = \mathrm{qmin\_factor} \times \min(k_1, k_2)` when :math:`k_1 = k_2`
        (the integrand is integrable there).

    Returns
    -------
    blocks : array (nells, nells, nk, nk)
        Or a dict of blocks ``{'T0': P^3 terms, 'SN_B': 1/nbar terms, 'SN_P': 1/nbar^2 terms}`` if ``return_terms``.
    """
    k = np.asarray(k, dtype=float)
    nk = len(k)
    ells = tuple(ells)
    if include_shotnoise and ndens is None:
        raise ValueError('ndens is required for the shot-noise trispectrum')
    if ndens is None: ndens = 1.
    if ndens2 is None: ndens2 = ndens**2
    kk, pp = _tabulate_pk(pk_lin, kedges)
    pk = _log_interpolator(kk, pp)
    kernels = _PTKernels(float(f), {name: float(value) for name, value in bias.items()})
    nlos, wlos = _los_grid(ntheta, nphi)
    nlos, wlos = jnp.array(nlos), jnp.array(wlos)
    xq, wq = _leggauss(nq)  # on [-1, 1], mapped to ln q below
    xq, wq = jnp.array(xq), jnp.array(wq)
    legendre = [get_legendre(ell) for ell in ells]

    def pair(k1, k2):
        # q = |k1 + k2| in [max(|k1 - k2|, qmin), k1 + k2], log-spaced Gauss-Legendre
        qmin = jnp.maximum(jnp.abs(k1 - k2), qmin_factor * jnp.minimum(k1, k2))
        qmax = k1 + k2
        lnq = 0.5 * (jnp.log(qmax) - jnp.log(qmin)) * xq + 0.5 * (jnp.log(qmax) + jnp.log(qmin))
        q = jnp.exp(lnq)
        # d mu12 / 2 = q dq / (2 k1 k2) = q^2 dlnq / (2 k1 k2)
        wmu = wq * 0.5 * (jnp.log(qmax) - jnp.log(qmin)) * q**2 / (2. * k1 * k2)
        mu12 = jnp.clip((q**2 - k1**2 - k2**2) / (2. * k1 * k2), -1., 1.)
        s12 = jnp.sqrt(1. - mu12**2)
        k1v = jnp.stack([jnp.zeros_like(q), jnp.zeros_like(q), jnp.full_like(q, k1)], axis=-1)[:, None, :]  # (nq, 1, 3)
        k2v = jnp.stack([k2 * s12, jnp.zeros_like(q), k2 * mu12], axis=-1)[:, None, :]
        n = nlos[None, :, :]  # (1, nlos, 3)
        qv = k1v + k2v
        # kernels
        Z1_1, Z1_2, Z1_q = kernels.Z1(k1v, n), kernels.Z1(k2v, n), kernels.Z1(qv, n)
        Z2_1q, Z2_2q = kernels.Z2(-k1v, qv, n), kernels.Z2(-k2v, qv, n)
        Z2_12 = kernels.Z2(k1v, k2v, n)
        Z3_112 = kernels.Z3(k1v, -k1v, k2v, n)
        Z3_221 = kernels.Z3(k2v, -k2v, k1v, n)
        P1, P2, Pq = pk(k1), pk(k2), pk(q)[:, None]
        T2211 = Pq * (8. * P1**2 * Z1_1**2 * Z2_1q**2 + 8. * P2**2 * Z1_2**2 * Z2_2q**2 + 16. * P1 * P2 * Z1_1 * Z1_2 * Z2_1q * Z2_2q)
        T3111 = 12. * (P1**2 * P2 * Z1_1**2 * Z1_2 * Z3_112 + P2**2 * P1 * Z1_2**2 * Z1_1 * Z3_221)
        SN_B = 8. * P1 * P2 * Z1_1 * Z1_2 * Z2_12 + 8. * Pq * Z1_q * (P1 * Z1_1 * Z2_1q + P2 * Z1_2 * Z2_2q)
        SN_P = 2. * Pq * Z1_q**2
        mu1, mu2 = kernels.mu(k1v, n), kernels.mu(k2v, n)
        L1 = jnp.stack([(2 * ell + 1) * leg(mu1) for ell, leg in zip(ells, legendre)])  # (nells, nq, nlos)
        L2 = jnp.stack([(2 * ell + 1) * leg(mu2) for ell, leg in zip(ells, legendre)])
        weight = wmu[:, None] * wlos[None, :]  # (nq, nlos)

        def integrate(T):
            return jnp.einsum('aqn,bqn,qn,qn->ab', L1, L2, weight, T)

        return jnp.stack([integrate(T2211 + T3111), integrate(SN_B), integrate(SN_P)])  # (3, nells, nells)

    ii, jj = np.triu_indices(nk)
    out = jax.lax.map(lambda kk: pair(kk[0], kk[1]), jnp.array(np.column_stack([k[ii], k[jj]])), batch_size=8)
    out = np.asarray(out)  # (npairs, 3, nells, nells)
    nells = len(ells)
    blocks = np.zeros((3, nells, nells, nk, nk))
    for p, (i, j) in enumerate(zip(ii, jj)):
        blocks[:, :, :, i, j] = out[p]
        blocks[:, :, :, j, i] = out[p].transpose(0, 2, 1)  # C_{l1 l2}(k1, k2) = C_{l2 l1}(k2, k1)
    terms = {'T0': blocks[0] / volume, 'SN_B': blocks[1] / (volume * ndens), 'SN_P': blocks[2] / (volume * ndens2)}
    if return_terms:
        return terms
    total = terms['T0']
    if include_shotnoise:
        total = total + terms['SN_B'] + terms['SN_P']
    return total


# ----------------------------------------------------------------------------------------------------------------------
# Super-sample covariance (CovaPT, Covariance_SurveyGeometry.ipynb, with the thecov corrections)
# ----------------------------------------------------------------------------------------------------------------------

def _Z12_kernel(ell, mu, dlnpk, b1, be, b2, g2):
    """Response kernel Z12 of CovaPT (Eq. A9 of Wadekar & Scoccimarro), for the multipole ``ell`` of the power spectrum."""
    if ell == 0:
        return (7 * b1**2 * be * (70 + 42 * be + (-35 * (-3 + dlnpk) + 3 * be * (28 + 13 * be - 14 * dlnpk - 5 * be * dlnpk)) * mu**2)
                + b1 * (35 * (47 - 7 * dlnpk) + be * (798 + 153 * be - 98 * dlnpk - 21 * be * dlnpk
                                                     + 4 * (84 + be * (48 - 21 * dlnpk) - 49 * dlnpk) * mu**2))
                + 98 * (5 * b2 * (3 + be) + 4 * g2 * (-5 + be * (-2 + mu**2)))) / (735. * b1**2)
    if ell == 2:
        return (14 * b1**2 * be * (14 + 12 * be + (2 * be * (12 + 7 * be) - (1 + be) * (7 + 5 * be) * dlnpk) * mu**2)
                + b1 * (4 * be * (69 + 19 * be) - 7 * be * (2 + be) * dlnpk
                        + (24 * be * (11 + 6 * be) - 7 * (21 + be * (22 + 9 * be)) * dlnpk) * mu**2 + 7 * (-8 + 7 * dlnpk + 24 * mu**2))
                + 28 * (7 * b2 * be + g2 * (-7 - 13 * be + (21 + 11 * be) * mu**2))) / (147. * b1**2)
    if ell == 4:
        return (8 * be * (b1 * (-132 + 77 * dlnpk + be * (23 + 154 * b1 + 14 * dlnpk)) - 154 * g2
                          + (b1 * (396 - 231 * dlnpk + be * (272 + 308 * b1 + 343 * b1 * be - 7 * (17 + b1 * (22 + 15 * be)) * dlnpk))
                             + 462 * g2) * mu**2)) / (2695. * b1**2)
    raise ValueError(ell)


def compute_ssc_covariance(k, pk_lin, f, bias, window_power, I10, I22, I32, kmax_window=None, ells=(0, 2, 4), dlnpk=None,
                           w10_multipoles=True, return_terms=False):
    r"""
    Super-sample covariance (beat coupling + local average) of :math:`P_{0, 2, 4}`, Eqs. (64)-(65), (77)-(78) of
    Wadekar & Scoccimarro (2020) as implemented in CovaPT's ``Covariance_SurveyGeometry.ipynb`` (``covaSSC``), with the
    corrections of the local ``thecov`` fork.

    Parameters
    ----------
    k : array (nk,)
    pk_lin : callable or array
        Linear **matter** power spectrum (a callable, or values at ``k`` together with ``dlnpk``).
    f : float
        Growth rate.
    bias : dict
        Needs ``b1``, ``b2``, ``bG2``.
    window_power : tuple (kavg, power)
        From :meth:`SurveyWindow.compute_window_power` (21 pairs, CovaPT normalization).
    I10, I22, I32 : float
        Window integrals (:meth:`SurveyWindow.I`).
    kmax_window : float, optional
        Upper limit of the :math:`\sigma^2` integrals (default: last ``kavg``).
    dlnpk : array, optional
        :math:`d \ln P / d \ln k` at ``k`` (computed by finite differences of ``pk_lin`` if callable).
    w10_multipoles : bool, default=True
        Include the quadrupole and hexadecapole of :math:`|W_{10}|^2` in :math:`\langle \delta n_g^2 \rangle`
        (CovaPT does, ``thecov`` only has the monopole).

    Returns
    -------
    blocks : array (nells, nells, nk, nk), or ``{'BC': ..., 'LA': ...}`` if ``return_terms``.
    """
    from scipy.integrate import quad
    from scipy.interpolate import InterpolatedUnivariateSpline

    k = np.asarray(k, dtype=float)
    nk = len(k)
    b1, b2, g2 = bias['b1'], bias['b2'], bias['bG2']
    be = f / b1
    if callable(pk_lin):
        pm = np.asarray(pk_lin(k), dtype=float)
        if dlnpk is None:
            dx = 1e-4 * k
            dlnpk = (np.asarray(pk_lin(k + dx)) - np.asarray(pk_lin(k - dx))) / (2 * dx) * k / pm
        pm_callable = pk_lin
    else:
        pm = np.asarray(pk_lin, dtype=float)
        if dlnpk is None:
            spl = InterpolatedUnivariateSpline(np.log(k), np.log(pm))
            dlnpk = spl.derivative()(np.log(k))
        pm_callable = InterpolatedUnivariateSpline(k, pm, ext='zeros')
    dlnpk = np.asarray(dlnpk, dtype=float)
    Plin = b1**2 * pm  # linear galaxy spectrum, as CovaPT

    kavg, power = window_power
    if kmax_window is None:
        kmax_window = kavg[-1]
    # sigma^2 = b1^2 / (2 pi^2) int dq q^2 P_m(q) P_win(q), for the 21 window pairs
    sigmas = np.array([b1**2 / (2 * np.pi**2) * quad(lambda q: q**2 * pm_callable(q) * InterpolatedUnivariateSpline(kavg, pw)(q), 0., kmax_window, limit=200)[0]
                       for pw in power])
    pairs = list(itertools.combinations_with_replacement(range(6), 2))
    index = {p: i for i, p in enumerate(pairs)}
    # multipoles: 0, 1, 2 = W22 L0, L2, L4; 3, 4, 5 = W10 L0, L2, L4
    sigma22x22 = np.array([[sigmas[index[tuple(sorted((i, j)))]] for j in range(3)] for i in range(3)])
    sigma22x10 = np.array([[sigmas[index[(i, 3 + j)]] for j in range(3)] for i in range(3)])
    sigma10x10 = np.array([[sigmas[index[tuple(sorted((3 + i, 3 + j)))]] for j in range(3)] for i in range(3)])

    ells = tuple(ells)
    nells = len(ells)
    xmu, wmu = _leggauss(64)
    # Z1[i] = int dmu L_{2i}(mu) (1 + be mu^2)
    Z1 = np.array([np.sum(wmu * _legendre_np(ell)(xmu) * (1 + be * xmu**2)) for ell in (0, 2, 4)])
    # Z12[l, i, k] = int dmu L_{2i}(mu) Z12_{ell_l}(mu; k)
    Z12 = np.array([[np.array([np.sum(wmu * _legendre_np(2 * i)(xmu) * _Z12_kernel(ell, xmu, d, b1, be, b2, g2)) for d in dlnpk])
                     for i in range(3)] for ell in ells])
    response = Plin[None, None, :] * Z12  # (nells, 3, nk)
    covBC = 1 / 4. * np.einsum('lik,ij,mjq->lmkq', response, sigma22x22, response)

    rsd = np.array([{0: 1 + 2 / 3 * be + 1 / 5 * be**2, 2: 4 / 3 * be + 4 / 7 * be**2, 4: 8 / 35 * be**2}[ell] for ell in ells])
    P_kaiser = rsd[:, None] * Plin[None, :]  # (nells, nk)
    # local-average term, Eq. (77)-(78): (1/4) sum_ij sigma22x10[i, j] Z12[l, i](k) Z1[j] + b2/b1^2 I32/(I22 I10) P_l(k) + 2/I10 rsd_l
    LA_term = 1 / 4. * np.einsum('lik,ij,j->lk', Z12, sigma22x10, Z1) + b2 * P_kaiser / b1**2 * I32 / I22 / I10 + 2. / I10 * rsd[:, None]
    # <delta n_g^2> = (1/4) sum_ij sigma10x10[i, j] Z1[i] Z1[j] + 1 / I10
    if w10_multipoles:
        S10 = 1 / 4. * np.einsum('ij,i,j->', sigma10x10, Z1, Z1) + 1. / I10
    else:
        S10 = 1 / 4. * sigma10x10[0, 0] * Z1[0]**2 + 1. / I10
    covLA = np.einsum('lk,mq->lmkq', P_kaiser, P_kaiser) * S10
    covLA -= np.einsum('lk,q,mq->lmkq', P_kaiser, Plin, LA_term)
    covLA -= np.einsum('lk,q,mq->mlqk', P_kaiser, Plin, LA_term)
    if return_terms:
        return {'BC': covBC, 'LA': covLA, 'sigma22x22': sigma22x22, 'sigma22x10': sigma22x10, 'sigma10x10': sigma10x10}
    return covBC + covLA
