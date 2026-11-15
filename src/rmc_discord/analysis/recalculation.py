r"""
Diffuse scattering recalculated from pair correlations.

For pair correlations :math:`C_{jj'}(\Delta\mathbf{R})` with pair counts
:math:`n_{jj'}(\Delta\mathbf{R})` (see :func:`correlations`), the diffuse
intensity per atom of the supercell is

.. math::

    I(\mathbf{Q}) = \frac{1}{n} \sum_{jj'\Delta\mathbf{R}}
        A_j^*(\mathbf{Q}) A_{j'}(\mathbf{Q})\,
        n_{jj'} w_{jj'}(\mathbf{Q})\, e^{i\mathbf{Q}\cdot\mathbf{r}}

with :math:`\mathbf{r} = \Delta\mathbf{R} + \mathbf{x}_{j'} - \mathbf{x}_j`.

- Occupancy (binary sites): :math:`A_j = P_k - P_{k'}`, the prefactor
  contrast of the correlated occupant and the other one (zero for a
  vacancy), and :math:`w = \langle\delta\delta\rangle`.
- Spins: :math:`A_j = M_j`, and :math:`w = \mathrm{tr}\,C -
  \hat{\mathbf{Q}}^T C \hat{\mathbf{Q}}` from the spin tensor
  :math:`C_{ab} = \langle S_a S_b\rangle` (moment perpendicular to Q).
- Displacements (with vacancies): the nuclear diffuse intensity for
  Gaussian pair displacements,

  .. math::

      I = \frac{1}{n} \sum P_j^* P_{j'} D_j D_{j'} e^{i\mathbf{Q}\cdot\mathbf{r}}
          \left[n_{jj'} e^{\mathbf{Q}^T C_{jj'} \mathbf{Q}} - N_c c_j c_{j'}\right]

  with the displacement tensor :math:`C_{jj'} = \langle\mathbf{u}\otimes
  \mathbf{u}\rangle` over occupied pairs (counts :math:`n_{jj'}`, so
  :math:`n_{jj'}/N_c = \langle o_j o_{j'}\rangle`), :math:`D_j =
  e^{-\mathbf{Q}^T U_j \mathbf{Q}/2}` from the on-site tensor, occupied
  fractions :math:`c_j`, and :math:`N_c` cells. The last term removes the
  Bragg part. The self term is exactly :math:`c - c^2 D^2`. Exact for
  jointly Gaussian pair displacements (all orders, not only the first) and
  for zero displacements; occupancy-displacement cross correlations (size
  effect) are not included.

Without truncation, and for one configuration, this equals the
configuration's diffuse intensity at its commensurate points (for spins when
each magnetic site has one magnetic species; otherwise the site prefactor is
the fraction-weighted mean). At other Q, and with ``r_max``, it is the
Fourier series of the (averaged, symmetrized, truncated) correlations.
"""

import numpy as np
from numba import njit, prange

from rmc_discord.model import scatterers


class _Points:
    """Minimal point set for the prefactor functions."""

    def __init__(self, Q):
        self.Q = np.ascontiguousarray(Q, dtype=float)
        self.Q_norm = np.linalg.norm(self.Q, axis=1)

    def __len__(self):
        return len(self.Q)


@njit(parallel=True, cache=True)
def _sum_scalar(Q, r, w, a, b, A, out):
    for q in prange(Q.shape[0]):
        acc = 0.0
        for p in range(r.shape[0]):
            ph = Q[q, 0] * r[p, 0] + Q[q, 1] * r[p, 1] + Q[q, 2] * r[p, 2]
            z = np.conj(A[a[p], q]) * A[b[p], q] * (np.cos(ph) + 1j * np.sin(ph))
            acc += w[p] * z.real
        out[q] = acc


@njit(parallel=True, cache=True)
def _sum_perp(Q, r, T, a, b, A, out):
    for q in prange(Q.shape[0]):
        n = np.sqrt(Q[q, 0] ** 2 + Q[q, 1] ** 2 + Q[q, 2] ** 2)
        h0, h1, h2 = Q[q, 0] / n, Q[q, 1] / n, Q[q, 2] / n
        acc = 0.0
        for p in range(r.shape[0]):
            t = T[p]
            qq = (
                h0 * (t[0, 0] * h0 + t[0, 1] * h1 + t[0, 2] * h2)
                + h1 * (t[1, 0] * h0 + t[1, 1] * h1 + t[1, 2] * h2)
                + h2 * (t[2, 0] * h0 + t[2, 1] * h1 + t[2, 2] * h2)
            )
            wq = t[0, 0] + t[1, 1] + t[2, 2] - qq
            ph = Q[q, 0] * r[p, 0] + Q[q, 1] * r[p, 1] + Q[q, 2] * r[p, 2]
            z = np.conj(A[a[p], q]) * A[b[p], q] * (np.cos(ph) + 1j * np.sin(ph))
            acc += wq * z.real
        out[q] = acc


@njit(parallel=True, cache=True)
def _sum_gauss(Q, r, T, cnt, cc, a, b, A, D, out):
    for q in prange(Q.shape[0]):
        q0, q1, q2 = Q[q, 0], Q[q, 1], Q[q, 2]
        acc = 0.0
        for p in range(r.shape[0]):
            t = T[p]
            qtq = (
                q0 * (t[0, 0] * q0 + t[0, 1] * q1 + t[0, 2] * q2)
                + q1 * (t[1, 0] * q0 + t[1, 1] * q1 + t[1, 2] * q2)
                + q2 * (t[2, 0] * q0 + t[2, 1] * q1 + t[2, 2] * q2)
            )
            wq = cnt[p] * np.exp(qtq) - cc[p]
            ph = q0 * r[p, 0] + q1 * r[p, 1] + q2 * r[p, 2]
            z = np.conj(A[a[p], q]) * A[b[p], q] * (np.cos(ph) + 1j * np.sin(ph))
            acc += wq * D[a[p], q] * D[b[p], q] * z.real
        out[q] = acc


def _site_prefactors(supercell, quantity, sites, Q, radiation, debye_waller):
    """Per-site amplitudes A_j(Q) for the sites in the table."""

    sc = supercell
    pts = _Points(Q)
    dw = range(sc.n_sites) if debye_waller else ()
    A = np.zeros((sc.n_sites, len(pts)), dtype=complex)
    if quantity == "occupancy":
        P = scatterers.nuclear_prefactor(sc, pts, radiation=radiation, debye_waller=dw)
        vacant = np.array([k.vacancy for k in sc.kinds])
        for j in sites:
            ks = list(sc.site_kinds[j])
            if len(ks) != 2:
                raise NotImplementedError("occupancy recalculation needs binary sites")
            k = next(k for k in ks if not vacant[k])
            other = ks[1] if ks[0] == k else ks[0]
            A[j] = P[k] - P[other]
    elif quantity == "displacement":
        P = scatterers.nuclear_prefactor(sc, pts, radiation=radiation, debye_waller=())
        vacant = np.array([k.vacancy for k in sc.kinds])
        for j in sites:
            occupied = [k for k in sc.site_kinds[j] if not vacant[k]]
            if len(occupied) != 1:
                raise NotImplementedError("displacement recalculation needs one occupant (plus vacancy) per site")
            A[j] = P[occupied[0]]
    elif quantity == "spin":
        M = scatterers.magnetic_prefactor(sc, pts, debye_waller=dw)
        for j in sites:
            ks = [k for k in sc.site_kinds[j] if sc.kinds[k].magnetic]
            f = np.array([sc.kinds[k].fraction for k in ks])
            A[j] = (f[:, None] * M[ks]).sum(axis=0) / f.sum()
    else:
        raise NotImplementedError(f"no recalculation for {quantity!r}")
    return A


def max_range(supercell):
    """
    Radius of the largest sphere inside the supercell (A).

    Pair vectors shorter than this are unambiguous minimum images and closed
    under the space group, so correlations truncated here can be evaluated at
    any Q; beyond it they stand for periodic images.
    """

    d = 1.0 / np.linalg.norm(supercell.crystal.B, axis=0)  # interplanar spacings d_100, d_010, d_001
    return float(np.min(np.array(supercell.N) * d) / 2)


def recalculate(supercell, corr, hkl, r_max=None, radiation="neutron", debye_waller=True):
    """
    Diffuse intensity from pair correlations at arbitrary Miller indices.

    Parameters
    ----------
    supercell : Supercell
        Geometry and kinds the correlations were computed for.
    corr : Correlations
        Occupancy, spin, or displacement correlations (unnormalized).
        Displacement correlations give the whole nuclear diffuse intensity
        (occupancy and displacements, Gaussian pair approximation).
    hkl : array_like, (n, 3)
        Miller indices (any, not only commensurate).
    r_max : float, optional
        Truncate the correlations at this distance (A). Without truncation
        the result is exact at the supercell's commensurate points; for other
        Q use ``r_max <= max_range(supercell)``.
    radiation : {"neutron", "xray"}, optional
        Nuclear prefactors for occupancy and displacements.
    debye_waller : bool, optional
        Include each kind's Debye-Waller factor in its prefactor (occupancy
        and spins; displacement recalculation uses the on-site tensor).

    Returns
    -------
    I : ndarray, (n,)
        Intensity per atom of the supercell (fm^2), on the refinement's scale.

    """

    if corr.normalized:
        raise ValueError("recalculation needs unnormalized correlations (normalize=False)")
    if corr.quantity not in ("occupancy", "spin", "displacement"):
        raise NotImplementedError(f"no recalculation for {corr.quantity!r}")
    c = corr if r_max is None else corr.within(r_max)
    hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
    Q = supercell.crystal.Q(hkl)
    sites = np.unique(c.sites)
    A = _site_prefactors(supercell, c.quantity, sites, Q, radiation, debye_waller)
    r = np.ascontiguousarray(c.r)
    a = np.ascontiguousarray(c.sites[:, 0], dtype=np.int64)
    b = np.ascontiguousarray(c.sites[:, 1], dtype=np.int64)
    out = np.zeros(len(Q))
    if c.quantity == "occupancy":
        w = np.ascontiguousarray(c.value * c.count)
        _sum_scalar(Q, r, w, a, b, A, out)
    elif c.quantity == "spin":
        T = np.ascontiguousarray(c.tensor * c.count[:, None, None])
        _sum_perp(Q, r, T, a, b, A, out)
    else:
        sc = supercell
        vacant = np.array([k.vacancy for k in sc.kinds])
        frac = np.zeros(sc.n_sites)
        for j in sites:
            frac[j] = sum(sc.kinds[k].fraction for k in sc.site_kinds[j] if not vacant[k])
        self_term = (corr.distance == 0) & (corr.sites[:, 0] == corr.sites[:, 1])
        U = np.zeros((sc.n_sites, 3, 3))
        U[corr.sites[self_term, 0]] = corr.tensor[self_term]
        D = np.exp(-0.5 * np.einsum("qi,jik,qk->jq", Q, U, Q))
        cc = np.ascontiguousarray(sc.n_cells * frac[a] * frac[b])
        _sum_gauss(Q, r, np.ascontiguousarray(c.tensor), np.ascontiguousarray(c.count), cc, a, b, A,
                   np.ascontiguousarray(D), out)
    return out / len(supercell)
