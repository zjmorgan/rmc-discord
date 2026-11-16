r"""Numba kernels for exact, incremental structure-factor updates.

Sample points are commensurate, :math:`\mathbf{h} = \mathbf{m}/\mathbf{N}`.
An atom in cell :math:`\mathbf{R}` on site :math:`j` with fractional
displacement :math:`\mathbf{u}` has phase

.. math::

    e^{i\mathbf{Q}\cdot\mathbf{r}} =
        \prod_a \omega_a[(m_a R_a) \bmod N_a] \;
        \Phi_j(\mathbf{Q}) \;
        e^{2\pi i \mathbf{h}\cdot\mathbf{u}}

with :math:`\omega_a[k] = e^{2\pi i k/N_a}` (tiny tables) and site phases
:math:`\Phi_j = e^{2\pi i \mathbf{h}\cdot\mathbf{x}_j}`. Trigonometric
functions are evaluated only for the displacement factor of atoms that are
displaced; the rest are table lookups and products. Displacements are kept
scaled, :math:`\mathbf{s} = 2\pi \mathbf{u}/\mathbf{N}`, so
:math:`2\pi\mathbf{h}\cdot\mathbf{u} = \mathbf{m}\cdot\mathbf{s}`.

Prefactors are factored as :math:`P_k(\mathbf{Q}) = c_k T_{t_k}(\mathbf{Q})`
with a per-kind constant :math:`c_k` and a shared Q-dependent table
:math:`T_t` (Debye-Waller factor, form factor, or group phase sum);
:math:`t_k = -1` means :math:`T = 1` and reads no memory.

For a move that changes atoms :math:`c` from (kind, position, spin)
:math:`(k, \mathbf{x}, \mathbf{S})` to :math:`(k', \mathbf{x}', \mathbf{S}')`

.. math::

    \Delta F = \sum_c P_{k'} e^{i\mathbf{Q}\cdot\mathbf{x}'}
             - P_{k} e^{i\mathbf{Q}\cdot\mathbf{x}}, \qquad
    \Delta \mathbf{F}_M = \sum_c M_{k'} \mathbf{S}' e^{i\mathbf{Q}\cdot\mathbf{x}'}
                        - M_{k} \mathbf{S} e^{i\mathbf{Q}\cdot\mathbf{x}}

with no Taylor expansion of the displacement phase. Evaluating a move only
accumulates the weighted sums needed for chi^2; the change is applied in a
second pass if the move is accepted.

Intensity per atom (fm^2) is

.. math::

    I = \frac{1}{n}\left(|F|^2 + |\mathbf{F}_M|^2
        - |\hat{\mathbf{Q}}\cdot\mathbf{F}_M|^2\right)

and the model compared with data is :math:`s (I - I_0) + b`, where
:math:`I_0` is an optional fixed offset (e.g. a subtracted paramagnetic
reference).

Argument groups
---------------
points : tuple
    ``(m, mm, B, w0, w1, w2, Phi)``: int16 indices and indices mod N
    (n_points, 3); ``2 pi B N^-1`` (3, 3) giving Cartesian Q from m; roots of
    unity per axis; site phases (n_sites, n_points).
data : tuple
    ``(w, wy, offset, has_offset, xbuf, has_basis)``: weights, weights times
    observations, model offset, whether the offset is nonzero, and a buffer
    that receives the candidate model term ``x = I - offset`` at every point
    when a background basis is fitted or points are averaged over
    symmetry equivalents.
basis : tuple
    ``(has_basis, kb, tb, Ginv, gy, yPy, partial)``: linear hat-function
    background in :math:`|Q|` (point q has weight ``1 - tb`` on function ``kb`` and
    ``tb`` on ``kb + 1``), with :math:`G = B^T W B`,
    :math:`g_y = G^{-1} B^T W y`, :math:`y^T P y = y^T W y - b_y^T g_y`, and
    a per-chunk work array.
factors : tuple
    ``(c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic)``.

Move kernels receive per-move lattice tables ``lat[c, a, k] =
omega_a[(k R_a) mod N_a]`` for the changed atoms' cells, so the per-point
loops contain no integer division.
"""

import numpy as np
from numba import njit, prange

FIT_FREE = 0
FIT_SCALE = 1
FIT_BACKGROUND = 2
FIT_NONE = 3

MOVE_SWAP = 0
MOVE_SPIN = 1
MOVE_DISPLACE = 2
MOVE_SWAP_DISPLACE = 3
MOVE_EXCHANGE = 4


@njit(cache=True)
def seed(value):
    """Seed Numba's random number generator."""
    np.random.seed(value)


@njit(cache=True)
def lattice_tables(points, cell, lat):
    """Fill ``lat[a, k] = omega_a[(k R_a) mod N_a]`` for one cell."""
    w0, w1, w2 = points[3], points[4], points[5]
    n0, n1, n2 = w0.shape[0], w1.shape[0], w2.shape[0]
    for k in range(n0):
        lat[0, k] = w0[(k * cell[0]) % n0]
    for k in range(n1):
        lat[1, k] = w1[(k * cell[1]) % n1]
    for k in range(n2):
        lat[2, k] = w2[(k * cell[2]) % n2]


@njit(cache=True, inline="always")
def _shift(m, q, s0, s1, s2):
    """Displacement phase exp(i m . s); exactly 1 for zero displacement."""
    if s0 == 0.0 and s1 == 0.0 and s2 == 0.0:
        return 1.0 + 0j
    p = m[q, 0] * s0 + m[q, 1] * s1 + m[q, 2] * s2
    return complex(np.cos(p), np.sin(p))


@njit(cache=True, inline="always")
def _nuc(c_nuc, t_nuc, T_nuc, k, q):
    t = t_nuc[k]
    if t < 0:
        return c_nuc[k]
    return c_nuc[k] * T_nuc[t, q]


@njit(cache=True, inline="always")
def _mag(c_mag, t_mag, T_mag, k, q):
    t = t_mag[k]
    if t < 0:
        return c_mag[k]
    return c_mag[k] * T_mag[t, q]


@njit(parallel=True, cache=True)
def structure_factors(points, factors, cell, site, su, kind, S, F, FM):
    """
    Full nuclear and magnetic structure factors.

    Parameters
    ----------
    points : tuple
        Sample point groups.
    factors : tuple
        Prefactor groups.
    cell : ndarray, (n, 3), int
        Cell of each atom.
    site : ndarray, (n,), int
        Active site of each atom.
    su : ndarray, (n, 3)
        Scaled displacements ``2 pi u / N`` (fractional ``u``).
    kind : ndarray, (n,), int
        Kind of each atom.
    S : ndarray, (n, 3)
        Spins.
    F : ndarray, complex, (n_points,)
        Output nuclear structure factor.
    FM : ndarray, complex, (n_points, 3)
        Output magnetic structure factor vector.

    """

    m, mm, B, w0, w1, w2, Phi = points
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    n = cell.shape[0]
    n0, n1, n2 = w0.shape[0], w1.shape[0], w2.shape[0]
    for q in prange(m.shape[0]):
        e0 = np.empty(n0, dtype=np.complex128)
        e1 = np.empty(n1, dtype=np.complex128)
        e2 = np.empty(n2, dtype=np.complex128)
        k = 0
        for r in range(n0):
            e0[r] = w0[k]
            k = (k + mm[q, 0]) % n0
        k = 0
        for r in range(n1):
            e1[r] = w1[k]
            k = (k + mm[q, 1]) % n1
        k = 0
        for r in range(n2):
            e2[r] = w2[k]
            k = (k + mm[q, 2]) % n2
        f = 0j
        fx = 0j
        fy = 0j
        fz = 0j
        for i in range(n):
            kk = kind[i]
            a = 0.0
            if magnetic:
                a = _mag(c_mag, t_mag, T_mag, kk, q)
            p = 0j
            if nuclear:
                p = _nuc(c_nuc, t_nuc, T_nuc, kk, q)
            if p == 0 and a == 0.0:
                continue
            e = e0[cell[i, 0]] * e1[cell[i, 1]] * e2[cell[i, 2]] * Phi[site[i], q]
            e = e * _shift(m, q, su[i, 0], su[i, 1], su[i, 2])
            f = f + p * e
            if a != 0.0:
                ae = a * e
                fx = fx + ae * S[i, 0]
                fy = fy + ae * S[i, 1]
                fz = fz + ae * S[i, 2]
        F[q] = f
        FM[q, 0] = fx
        FM[q, 1] = fy
        FM[q, 2] = fz


@njit(parallel=True, cache=True)
def atom_tables(m_lo, theta, E):
    """``E[a, k, i] = exp(i (m_lo[a] + k) theta[i, a])`` for a block of atoms."""
    n = theta.shape[0]
    L = E.shape[1]
    for i in prange(n):
        for a in range(3):
            th = theta[i, a]
            step = complex(np.cos(th), np.sin(th))
            z = 0j
            for k in range(L):
                if k % 32 == 0:
                    p = (m_lo[a] + k) * th
                    z = complex(np.cos(p), np.sin(p))
                E[a, k, i] = z
                z = z * step


@njit(parallel=True, cache=True)
def accumulate_atoms(m, m_lo, E, n, cn, cm, S, T_nuc, has_tn, T_mag, has_tm, nuclear, magnetic, F, FM):
    r"""
    Add a block of ``n`` atoms sharing prefactor tables to the structure
    factors.

    Atom ``i`` has phase :math:`\prod_a E[a, m_a - m_{lo,a}, i]`, nuclear
    constant ``cn[i]`` times ``T_nuc`` (if ``has_tn``), and magnetic constant
    ``cm[i]`` times ``T_mag`` (if ``has_tm``) along spin ``S[i]``.
    """
    for q in prange(m.shape[0]):
        r0 = E[0, m[q, 0] - m_lo[0]]
        r1 = E[1, m[q, 1] - m_lo[1]]
        r2 = E[2, m[q, 2] - m_lo[2]]
        f = 0j
        fx = 0j
        fy = 0j
        fz = 0j
        if magnetic:
            for i in range(n):
                e = r0[i] * r1[i] * r2[i]
                if nuclear:
                    f += cn[i] * e
                ae = cm[i] * e
                fx += ae * S[i, 0]
                fy += ae * S[i, 1]
                fz += ae * S[i, 2]
        else:
            for i in range(n):
                f += cn[i] * (r0[i] * r1[i] * r2[i])
        if nuclear:
            if has_tn:
                F[q] += f * T_nuc[q]
            else:
                F[q] += f
        if magnetic:
            t = T_mag[q] if has_tm else 1.0
            FM[q, 0] += t * fx
            FM[q, 1] += t * fy
            FM[q, 2] += t * fz


@njit(cache=True, inline="always")
def _I(m, B, q, f, fx, fy, fz, magnetic):
    """Unnormalized intensity |F|^2 + |F_M,perp|^2 at point q."""
    I = f.real * f.real + f.imag * f.imag
    if magnetic:
        h0 = m[q, 0]
        h1 = m[q, 1]
        h2 = m[q, 2]
        Qx = B[0, 0] * h0 + B[0, 1] * h1 + B[0, 2] * h2
        Qy = B[1, 0] * h0 + B[1, 1] * h1 + B[1, 2] * h2
        Qz = B[2, 0] * h0 + B[2, 1] * h1 + B[2, 2] * h2
        p = Qx * fx + Qy * fy + Qz * fz
        Q2 = Qx * Qx + Qy * Qy + Qz * Qz
        I += (
            fx.real * fx.real
            + fx.imag * fx.imag
            + fy.real * fy.real
            + fy.imag * fy.imag
            + fz.real * fz.real
            + fz.imag * fz.imag
            - (p.real * p.real + p.imag * p.imag) / Q2
        )
    return I


@njit(cache=True, inline="always")
def _base(lat, c, mm, q, Phi, site):
    """Phase of an undisplaced atom from its lattice table and site phase."""
    return lat[c, 0, mm[q, 0]] * lat[c, 1, mm[q, 1]] * lat[c, 2, mm[q, 2]] * Phi[site, q]


@njit(parallel=True, cache=True)
def intensity(points, F, FM, inv_n, magnetic, out):
    """Intensity per atom from the structure factors."""
    m, B = points[0], points[2]
    for q in prange(F.shape[0]):
        out[q] = inv_n * _I(m, B, q, F[q], FM[q, 0], FM[q, 1], FM[q, 2], magnetic)


@njit(parallel=True, cache=True)
def sums(I, data):
    """Weighted sums of the model term ``x = I - offset``."""
    w, wy, offset, has_offset, xbuf, has_basis = data
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for q in prange(I.shape[0]):
        xq = I[q]
        if has_offset:
            xq -= offset[q]
        if has_basis:
            xbuf[q] = xq
        wx = w[q] * xq
        sx += wx
        sxx += wx * xq
        sxy += wy[q] * xq
    return sx, sxx, sxy


@njit(cache=True)
def fit(sw, sy, syy, sx, sxx, sxy, mode, scale, background):
    r"""
    Closed-form scale and background minimizing chi^2.

    Minimizes :math:`\sum_q w_q (s x_q + b - y_q)^2` over the free
    parameters given the weighted sums. The scale is kept nonnegative.

    Parameters
    ----------
    sw, sy, syy : float
        :math:`\sum w`, :math:`\sum w y`, :math:`\sum w y^2`.
    sx, sxx, sxy : float
        :math:`\sum w x`, :math:`\sum w x^2`, :math:`\sum w x y`.
    mode : int
        ``FIT_FREE``, ``FIT_SCALE`` (background fixed), ``FIT_BACKGROUND``
        (scale fixed), or ``FIT_NONE``.
    scale, background : float
        Fixed values for parameters that are not fitted.

    Returns
    -------
    chi_sq : float
        Total chi^2.
    scale, background : float
        Optimal (or fixed) values.

    """

    s = scale
    b = background
    if mode == FIT_FREE:
        det = sw * sxx - sx * sx
        if det > 0.0:
            s = (sw * sxy - sx * sy) / det
        else:
            s = 0.0
        if s < 0.0:
            s = 0.0
        b = (sy - s * sx) / sw
    elif mode == FIT_SCALE:
        s = (sxy - b * sx) / sxx if sxx > 0.0 else 0.0
        if s < 0.0:
            s = 0.0
    elif mode == FIT_BACKGROUND:
        b = (sy - s * sx) / sw
    chi_sq = syy - 2 * s * sxy - 2 * b * sy + s * s * sxx + 2 * s * b * sx + b * b * sw
    return max(chi_sq, 0.0), s, b


@njit(parallel=True, cache=True)
def basis_sums(x, w, kb, tb, partial):
    """:math:`B^T W x` for a hat-function basis, by chunks."""
    n = x.shape[0]
    n_chunks, K = partial.shape
    for c in prange(n_chunks):
        for k in range(K):
            partial[c, k] = 0.0
        for q in range(c * n // n_chunks, (c + 1) * n // n_chunks):
            v = w[q] * x[q]
            k = kb[q]
            t = tb[q]
            partial[c, k] += v * (1.0 - t)
            partial[c, k + 1] += v * t
    bx = np.zeros(K)
    for c in range(n_chunks):
        for k in range(K):
            bx[k] += partial[c, k]
    return bx


@njit(cache=True)
def fit_basis(sxx, sxy, bx, Ginv, gy, yPy, mode, scale):
    r"""
    Closed-form scale with a background basis eliminated.

    With :math:`P = W - W B G^{-1} B^T W`,

    .. math::

        \chi^2(s) = y^T P y - 2 s\, x^T P y + s^2\, x^T P x,

    :math:`x^T P y = \sum w x y - b_x^T g_y` and
    :math:`x^T P x = \sum w x^2 - b_x^T G^{-1} b_x`. The scale is kept
    nonnegative; ``FIT_BACKGROUND`` keeps it fixed.

    Returns
    -------
    chi_sq, scale : float

    """

    K = bx.shape[0]
    xPy = sxy
    xPx = sxx
    for i in range(K):
        xPy -= bx[i] * gy[i]
        v = 0.0
        for j in range(K):
            v += Ginv[i, j] * bx[j]
        xPx -= bx[i] * v
    s = scale
    if mode == FIT_FREE:
        s = xPy / xPx if xPx > 0.0 else 0.0
        if s < 0.0:
            s = 0.0
    chi_sq = yPy - 2.0 * s * xPy + s * s * xPx
    return max(chi_sq, 0.0), s


@njit(parallel=True, cache=True)
def obs_sums(xbuf, n_ops, w, wy, xobs):
    """
    Weighted sums of observations averaged over symmetry-equivalent points.

    Point ``o + r n`` is the ``r``-th equivalent of observation ``o``; the
    averaged model term is written to ``xobs``.
    """
    n = xobs.shape[0]
    inv = 1.0 / n_ops
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for o in prange(n):
        v = 0.0
        for r in range(n_ops):
            v += xbuf[o + r * n]
        v *= inv
        xobs[o] = v
        wx = w[o] * v
        sx += wx
        sxx += wx * v
        sxy += wy[o] * v
    return sx, sxx, sxy


@njit(cache=True)
def _observed(sx, sxx, sxy, data, obs):
    """Point sums, or observation sums when points are averaged."""
    if obs[0] > 1:
        return obs_sums(data[4], obs[0], obs[1], obs[2], obs[3])
    return sx, sxx, sxy


@njit(cache=True)
def _fit(sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background, obs, basis):
    """Fit with the scalar background or the basis."""
    if basis[0]:
        bx = basis_sums(obs[3], obs[1], basis[1], basis[2], basis[6])
        chi_sq, s = fit_basis(sxx, sxy, bx, basis[3], basis[4], basis[5], fit_mode, scale)
        return chi_sq, s, 0.0
    return fit(sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background)


# ----------------------------------------------------------------------
# Move kernels. Each move type has an evaluate kernel (weighted sums of the
# candidate) and an apply kernel (add the change). Tuples are unpacked before
# the parallel loop; per-point helpers take only arrays and scalars. The
# candidate buffer ``xbuf`` is a separate argument: in Numba 0.68, stores in a
# prange loop that calls an inlined tuple-returning helper are silently lost
# when the array was unpacked from a tuple argument.


@njit(cache=True, inline="always")
def _accumulate(m, B, q, f, fx, fy, fz, magnetic, inv_n, w, wy, offset, has_offset):
    I = inv_n * _I(m, B, q, f, fx, fy, fz, magnetic)
    if has_offset:
        I -= offset[q]
    wx = w[q] * I
    return wx, wx * I, wy[q] * I, I


@njit(parallel=True, cache=True)
def evaluate_spin(points, data, xbuf, factors, F, FM, inv_n, lat, site, su, k, dS):
    r"""
    Candidate sums for rotating one spin: :math:`\Delta\mathbf{F}_M =
    M_k e^{i\mathbf{Q}\cdot\mathbf{r}} (\mathbf{S}' - \mathbf{S})`.
    """
    m, mm, B, w0, w1, w2, Phi = points
    w, wy, offset, has_offset, _, has_basis = data
    c_mag, t_mag, T_mag = factors[3], factors[4], factors[5]
    s0, s1, s2 = su[0], su[1], su[2]
    d0, d1, d2 = dS[0], dS[1], dS[2]
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for q in prange(F.shape[0]):
        e = _base(lat, 0, mm, q, Phi, site) * _shift(m, q, s0, s1, s2)
        a = _mag(c_mag, t_mag, T_mag, k, q) * e
        x0, x1, x2, xq = _accumulate(
            m, B, q, F[q], FM[q, 0] + a * d0, FM[q, 1] + a * d1, FM[q, 2] + a * d2,
            True, inv_n, w, wy, offset, has_offset,
        )
        if has_basis:
            xbuf[q] = xq
        sx += x0
        sxx += x1
        sxy += x2
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_spin(points, factors, FM, lat, site, su, k, dS):
    m, mm, B, w0, w1, w2, Phi = points
    c_mag, t_mag, T_mag = factors[3], factors[4], factors[5]
    s0, s1, s2 = su[0], su[1], su[2]
    d0, d1, d2 = dS[0], dS[1], dS[2]
    for q in prange(FM.shape[0]):
        e = _base(lat, 0, mm, q, Phi, site) * _shift(m, q, s0, s1, s2)
        a = _mag(c_mag, t_mag, T_mag, k, q) * e
        FM[q, 0] += a * d0
        FM[q, 1] += a * d1
        FM[q, 2] += a * d2


@njit(parallel=True, cache=True)
def evaluate_displace(points, data, xbuf, factors, F, FM, inv_n, lat, site, su_old, su_new, k, S):
    r"""
    Candidate sums for displacing one atom: with
    :math:`\Delta e = e^{i\mathbf{Q}\cdot\mathbf{r}'} - e^{i\mathbf{Q}\cdot\mathbf{r}}`,
    :math:`\Delta F = P_k \Delta e` and
    :math:`\Delta\mathbf{F}_M = M_k \mathbf{S} \Delta e`.
    """
    m, mm, B, w0, w1, w2, Phi = points
    w, wy, offset, has_offset, _, has_basis = data
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    o0, o1, o2 = su_old[0], su_old[1], su_old[2]
    n0, n1, n2 = su_new[0], su_new[1], su_new[2]
    S0, S1, S2 = S[0], S[1], S[2]
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for q in prange(F.shape[0]):
        de = _base(lat, 0, mm, q, Phi, site) * (
            _shift(m, q, n0, n1, n2) - _shift(m, q, o0, o1, o2)
        )
        f = F[q]
        fx = FM[q, 0]
        fy = FM[q, 1]
        fz = FM[q, 2]
        if nuclear:
            f = f + _nuc(c_nuc, t_nuc, T_nuc, k, q) * de
        if magnetic:
            a = _mag(c_mag, t_mag, T_mag, k, q) * de
            fx = fx + a * S0
            fy = fy + a * S1
            fz = fz + a * S2
        x0, x1, x2, xq = _accumulate(
            m, B, q, f, fx, fy, fz, magnetic, inv_n, w, wy, offset, has_offset
        )
        if has_basis:
            xbuf[q] = xq
        sx += x0
        sxx += x1
        sxy += x2
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_displace(points, factors, F, FM, lat, site, su_old, su_new, k, S):
    m, mm, B, w0, w1, w2, Phi = points
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    o0, o1, o2 = su_old[0], su_old[1], su_old[2]
    n0, n1, n2 = su_new[0], su_new[1], su_new[2]
    S0, S1, S2 = S[0], S[1], S[2]
    for q in prange(F.shape[0]):
        de = _base(lat, 0, mm, q, Phi, site) * (
            _shift(m, q, n0, n1, n2) - _shift(m, q, o0, o1, o2)
        )
        if nuclear:
            F[q] += _nuc(c_nuc, t_nuc, T_nuc, k, q) * de
        if magnetic:
            a = _mag(c_mag, t_mag, T_mag, k, q) * de
            FM[q, 0] += a * S0
            FM[q, 1] += a * S1
            FM[q, 2] += a * S2


@njit(parallel=True, cache=True)
def evaluate_swap(points, data, xbuf, factors, F, FM, inv_n, lat, sites, su, ka, kb, Sa, Sb):
    r"""
    Candidate sums for exchanging the kinds of atoms a and b:
    :math:`\Delta F = (P_b - P_a)(e_a - e_b)` and
    :math:`\Delta\mathbf{F}_M = (M_b - M_a)(\mathbf{S}_a e_a - \mathbf{S}_b e_b)`.
    """
    m, mm, B, w0, w1, w2, Phi = points
    w, wy, offset, has_offset, _, has_basis = data
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    ja, jb = sites[0], sites[1]
    a0, a1, a2 = su[0, 0], su[0, 1], su[0, 2]
    b0, b1, b2 = su[1, 0], su[1, 1], su[1, 2]
    Sa0, Sa1, Sa2 = Sa[0], Sa[1], Sa[2]
    Sb0, Sb1, Sb2 = Sb[0], Sb[1], Sb[2]
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for q in prange(F.shape[0]):
        ea = _base(lat, 0, mm, q, Phi, ja) * _shift(m, q, a0, a1, a2)
        eb = _base(lat, 1, mm, q, Phi, jb) * _shift(m, q, b0, b1, b2)
        f = F[q]
        fx = FM[q, 0]
        fy = FM[q, 1]
        fz = FM[q, 2]
        if nuclear:
            dP = _nuc(c_nuc, t_nuc, T_nuc, kb, q) - _nuc(c_nuc, t_nuc, T_nuc, ka, q)
            f = f + dP * (ea - eb)
        if magnetic:
            dM = _mag(c_mag, t_mag, T_mag, kb, q) - _mag(c_mag, t_mag, T_mag, ka, q)
            fx = fx + dM * (Sa0 * ea - Sb0 * eb)
            fy = fy + dM * (Sa1 * ea - Sb1 * eb)
            fz = fz + dM * (Sa2 * ea - Sb2 * eb)
        x0, x1, x2, xq = _accumulate(
            m, B, q, f, fx, fy, fz, magnetic, inv_n, w, wy, offset, has_offset
        )
        if has_basis:
            xbuf[q] = xq
        sx += x0
        sxx += x1
        sxy += x2
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_swap(points, factors, F, FM, lat, sites, su, ka, kb, Sa, Sb):
    m, mm, B, w0, w1, w2, Phi = points
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    ja, jb = sites[0], sites[1]
    a0, a1, a2 = su[0, 0], su[0, 1], su[0, 2]
    b0, b1, b2 = su[1, 0], su[1, 1], su[1, 2]
    Sa0, Sa1, Sa2 = Sa[0], Sa[1], Sa[2]
    Sb0, Sb1, Sb2 = Sb[0], Sb[1], Sb[2]
    for q in prange(F.shape[0]):
        ea = _base(lat, 0, mm, q, Phi, ja) * _shift(m, q, a0, a1, a2)
        eb = _base(lat, 1, mm, q, Phi, jb) * _shift(m, q, b0, b1, b2)
        if nuclear:
            dP = _nuc(c_nuc, t_nuc, T_nuc, kb, q) - _nuc(c_nuc, t_nuc, T_nuc, ka, q)
            F[q] += dP * (ea - eb)
        if magnetic:
            dM = _mag(c_mag, t_mag, T_mag, kb, q) - _mag(c_mag, t_mag, T_mag, ka, q)
            FM[q, 0] += dM * (Sa0 * ea - Sb0 * eb)
            FM[q, 1] += dM * (Sa1 * ea - Sb1 * eb)
            FM[q, 2] += dM * (Sa2 * ea - Sb2 * eb)


@njit(parallel=True, cache=True)
def evaluate_pair(points, data, xbuf, factors, F, FM, inv_n, lat, sites, su_old, su_new, k_old, k_new, S):
    r"""
    Candidate sums for changing the kinds and displacements of atoms a and b,
    :math:`\Delta F = \sum_c P_{k'_c} e_c(\mathbf{s}'_c) - P_{k_c} e_c(\mathbf{s}_c)`,
    with spins kept.
    """
    m, mm, B, w0, w1, w2, Phi = points
    w, wy, offset, has_offset, _, has_basis = data
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    ja, jb = sites[0], sites[1]
    ka, kb, la, lb = k_old[0], k_old[1], k_new[0], k_new[1]
    a0, a1, a2 = su_old[0, 0], su_old[0, 1], su_old[0, 2]
    b0, b1, b2 = su_old[1, 0], su_old[1, 1], su_old[1, 2]
    c0, c1, c2 = su_new[0, 0], su_new[0, 1], su_new[0, 2]
    d0, d1, d2 = su_new[1, 0], su_new[1, 1], su_new[1, 2]
    Sa0, Sa1, Sa2 = S[0, 0], S[0, 1], S[0, 2]
    Sb0, Sb1, Sb2 = S[1, 0], S[1, 1], S[1, 2]
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for q in prange(F.shape[0]):
        ea = _base(lat, 0, mm, q, Phi, ja)
        eb = _base(lat, 1, mm, q, Phi, jb)
        ea_old = ea * _shift(m, q, a0, a1, a2)
        eb_old = eb * _shift(m, q, b0, b1, b2)
        ea_new = ea * _shift(m, q, c0, c1, c2)
        eb_new = eb * _shift(m, q, d0, d1, d2)
        f = F[q]
        fx = FM[q, 0]
        fy = FM[q, 1]
        fz = FM[q, 2]
        if nuclear:
            f = (
                f
                + _nuc(c_nuc, t_nuc, T_nuc, la, q) * ea_new
                - _nuc(c_nuc, t_nuc, T_nuc, ka, q) * ea_old
                + _nuc(c_nuc, t_nuc, T_nuc, lb, q) * eb_new
                - _nuc(c_nuc, t_nuc, T_nuc, kb, q) * eb_old
            )
        if magnetic:
            ga = _mag(c_mag, t_mag, T_mag, la, q) * ea_new - _mag(c_mag, t_mag, T_mag, ka, q) * ea_old
            gb = _mag(c_mag, t_mag, T_mag, lb, q) * eb_new - _mag(c_mag, t_mag, T_mag, kb, q) * eb_old
            fx = fx + Sa0 * ga + Sb0 * gb
            fy = fy + Sa1 * ga + Sb1 * gb
            fz = fz + Sa2 * ga + Sb2 * gb
        x0, x1, x2, xq = _accumulate(
            m, B, q, f, fx, fy, fz, magnetic, inv_n, w, wy, offset, has_offset
        )
        if has_basis:
            xbuf[q] = xq
        sx += x0
        sxx += x1
        sxy += x2
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_pair(points, factors, F, FM, lat, sites, su_old, su_new, k_old, k_new, S):
    m, mm, B, w0, w1, w2, Phi = points
    c_nuc, t_nuc, T_nuc, c_mag, t_mag, T_mag, nuclear, magnetic = factors
    ja, jb = sites[0], sites[1]
    ka, kb, la, lb = k_old[0], k_old[1], k_new[0], k_new[1]
    a0, a1, a2 = su_old[0, 0], su_old[0, 1], su_old[0, 2]
    b0, b1, b2 = su_old[1, 0], su_old[1, 1], su_old[1, 2]
    c0, c1, c2 = su_new[0, 0], su_new[0, 1], su_new[0, 2]
    d0, d1, d2 = su_new[1, 0], su_new[1, 1], su_new[1, 2]
    Sa0, Sa1, Sa2 = S[0, 0], S[0, 1], S[0, 2]
    Sb0, Sb1, Sb2 = S[1, 0], S[1, 1], S[1, 2]
    for q in prange(F.shape[0]):
        ea = _base(lat, 0, mm, q, Phi, ja)
        eb = _base(lat, 1, mm, q, Phi, jb)
        ea_old = ea * _shift(m, q, a0, a1, a2)
        eb_old = eb * _shift(m, q, b0, b1, b2)
        ea_new = ea * _shift(m, q, c0, c1, c2)
        eb_new = eb * _shift(m, q, d0, d1, d2)
        if nuclear:
            F[q] += (
                _nuc(c_nuc, t_nuc, T_nuc, la, q) * ea_new
                - _nuc(c_nuc, t_nuc, T_nuc, ka, q) * ea_old
                + _nuc(c_nuc, t_nuc, T_nuc, lb, q) * eb_new
                - _nuc(c_nuc, t_nuc, T_nuc, kb, q) * eb_old
            )
        if magnetic:
            ga = _mag(c_mag, t_mag, T_mag, la, q) * ea_new - _mag(c_mag, t_mag, T_mag, ka, q) * ea_old
            gb = _mag(c_mag, t_mag, T_mag, lb, q) * eb_new - _mag(c_mag, t_mag, T_mag, kb, q) * eb_old
            FM[q, 0] += Sa0 * ga + Sb0 * gb
            FM[q, 1] += Sa1 * ga + Sb1 * gb
            FM[q, 2] += Sa2 * ga + Sb2 * gb


# ----------------------------------------------------------------------
# Fused nuclear move kernels. Without magnetic scattering a move changes
# F by a few terms, each an atom's phase times a prefactor,
#
#     dF(Q) = sum_t (c_t1 T_t1(Q) + c_t2 T_t2(Q)) prod_a E_ta[m_a],
#
# with per-move phase tables E_ta[k] = exp(i k theta_a), theta_a =
# 2 pi (R_a + x_a)/N_a + s_a, over the range of m_a. A point costs three
# table lookups per term and no trigonometric function. Evaluation loops
# over observations, so averaging symmetry equivalents and the background
# basis sums happen in the same pass, by chunks.

MAX_TERMS = 4


@njit(cache=True)
def phase_table(m_lo, theta, E):
    """``E[a, k] = exp(i (m_lo[a] + k) theta[a])`` by recurrence, reset every 32."""
    L = E.shape[1]
    for a in range(3):
        th = theta[a]
        step = complex(np.cos(th), np.sin(th))
        z = 0j
        for k in range(L):
            if k % 32 == 0:
                p = (m_lo[a] + k) * th
                z = complex(np.cos(p), np.sin(p))
            E[a, k] = z
            z = z * step


@njit(cache=True, inline="always")
def _delta_nuc(m, q, m_lo, E, n_terms, coef, tt, T):
    d = 0j
    k0 = m[q, 0] - m_lo[0]
    k1 = m[q, 1] - m_lo[1]
    k2 = m[q, 2] - m_lo[2]
    for t in range(n_terms):
        p = coef[t, 0] if tt[t, 0] < 0 else coef[t, 0] * T[tt[t, 0], q]
        if coef[t, 1] != 0:
            p += coef[t, 1] if tt[t, 1] < 0 else coef[t, 1] * T[tt[t, 1], q]
        d += p * E[t, 0, k0] * E[t, 1, k1] * E[t, 2, k2]
    return d


@njit(parallel=True, cache=True)
def evaluate_nuclear(m, F, T, m_lo, E, n_terms, coef, tt, inv_n, n_ops,
                     w, wy, offset, has_offset, kb, tb, has_basis, partial, chunk_sums):
    r"""
    Candidate observation sums of a nuclear move.

    Returns :math:`\sum w x`, :math:`\sum w x^2`, :math:`\sum w x y` of the
    symmetry-averaged candidate model ``x`` and, with a basis, the chunk
    partials of :math:`B^T W x` in ``partial``.
    """
    n_obs = w.shape[0]
    n_chunks = partial.shape[0]
    K = partial.shape[1]
    inv_ops = 1.0 / n_ops
    for c in prange(n_chunks):
        if has_basis:
            for k in range(K):
                partial[c, k] = 0.0
        sx = 0.0
        sxx = 0.0
        sxy = 0.0
        for o in range(c * n_obs // n_chunks, (c + 1) * n_obs // n_chunks):
            v = 0.0
            for r in range(n_ops):
                q = o + r * n_obs
                f = F[q] + _delta_nuc(m, q, m_lo, E, n_terms, coef, tt, T)
                x = inv_n * (f.real * f.real + f.imag * f.imag)
                if has_offset:
                    x -= offset[q]
                v += x
            v *= inv_ops
            wx = w[o] * v
            sx += wx
            sxx += wx * v
            sxy += wy[o] * v
            if has_basis:
                partial[c, kb[o]] += wx * (1.0 - tb[o])
                partial[c, kb[o] + 1] += wx * tb[o]
        chunk_sums[c, 0] = sx
        chunk_sums[c, 1] = sxx
        chunk_sums[c, 2] = sxy
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for c in range(n_chunks):
        sx += chunk_sums[c, 0]
        sxx += chunk_sums[c, 1]
        sxy += chunk_sums[c, 2]
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_nuclear(m, F, T, m_lo, E, n_terms, coef, tt):
    for q in prange(F.shape[0]):
        F[q] += _delta_nuc(m, q, m_lo, E, n_terms, coef, tt, T)


@njit(cache=True)
def _fit_nuclear(sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background, basis, partial):
    if basis[0]:
        K = partial.shape[1]
        bx = np.zeros(K)
        for c in range(partial.shape[0]):
            for k in range(K):
                bx[k] += partial[c, k]
        chi_sq, s = fit_basis(sxx, sxy, bx, basis[3], basis[4], basis[5], fit_mode, scale)
        return chi_sq, s, 0.0
    return fit(sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background)


@njit(cache=True)
def _add_term(n, cell, xyz, su, tpN, m_lo, E, coef, tt, mcoef, mtt, vec,
              c1, t1, c2, t2, a1, u1, a2, u2, v, theta):
    """
    Append a term: an atom phase table times the nuclear coefficient
    ``c1 T_t1 + c2 T_t2`` and the magnetic one ``(a1 T_u1 + a2 T_u2) v``,
    unless both are zero.
    """
    mag = (a1 != 0 or a2 != 0) and (v[0] != 0 or v[1] != 0 or v[2] != 0)
    if c1 == 0 and c2 == 0 and not mag:
        return n
    if c1 == 0:
        c1, t1, c2, t2 = c2, t2, 0j, -1
    if a1 == 0:
        a1, u1, a2, u2 = a2, u2, 0.0, -1
    for a in range(3):
        theta[a] = tpN[a] * (cell[a] + xyz[a]) + su[a]
    phase_table(m_lo, theta, E[n])
    coef[n, 0] = c1
    coef[n, 1] = c2
    tt[n, 0] = t1
    tt[n, 1] = t2
    if mag:
        mcoef[n, 0] = a1
        mcoef[n, 1] = a2
        mtt[n, 0] = u1
        mtt[n, 1] = u2
    else:
        mcoef[n, 0] = 0.0
        mcoef[n, 1] = 0.0
        mtt[n, 0] = -1
        mtt[n, 1] = -1
    for e in range(3):
        vec[n, e] = v[e]
    return n + 1


@njit(cache=True)
def _move_terms(move, n_change, idx, cell, xyz, c_site, c_su_old, c_su_new, k_old, k_new,
                S_old, dS, factors, tpN, m_lo, E, coef, tt, mcoef, mtt, vec, theta):
    """Terms of a move for the fused kernels; returns their number."""
    c_nuc, t_nuc, c_mag, t_mag = factors[0], factors[1], factors[3], factors[4]
    nuclear, magnetic = factors[6], factors[7]
    n = 0
    if move == MOVE_SPIN:
        k = k_old[0]
        a = c_mag[k] if magnetic else 0.0
        return _add_term(n, cell[idx[0]], xyz[c_site[0]], c_su_old[0], tpN, m_lo, E, coef, tt, mcoef, mtt,
                         vec, 0j, -1, 0j, -1, a, t_mag[k], 0.0, -1, dS, theta)
    if move == MOVE_SWAP:
        for c in range(2):
            ka, kb = k_old[c], k_old[1 - c]
            cn_a = c_nuc[ka] if nuclear else 0j
            cn_b = c_nuc[kb] if nuclear else 0j
            cm_a = c_mag[ka] if magnetic else 0.0
            cm_b = c_mag[kb] if magnetic else 0.0
            n = _add_term(n, cell[idx[c]], xyz[c_site[c]], c_su_old[c], tpN, m_lo, E, coef, tt, mcoef, mtt,
                          vec, cn_b, t_nuc[kb], -cn_a, t_nuc[ka], cm_b, t_mag[kb], -cm_a, t_mag[ka],
                          S_old[c], theta)
        return n
    for c in range(n_change):
        i = idx[c]
        for side in range(2):
            k = k_new[c] if side == 0 else k_old[c]
            sign = 1.0 if side == 0 else -1.0
            su = c_su_new[c] if side == 0 else c_su_old[c]
            cn = sign * c_nuc[k] if nuclear else 0j
            cm = sign * c_mag[k] if magnetic else 0.0
            n = _add_term(n, cell[i], xyz[c_site[c]], su, tpN, m_lo, E, coef, tt, mcoef, mtt, vec,
                          cn, t_nuc[k], 0j, -1, cm, t_mag[k], 0.0, -1, S_old[c], theta)
    return n


@njit(cache=True, inline="always")
def _delta_mag(m, q, m_lo, E, n_terms, coef, tt, T, mcoef, mtt, Tm, vec, nuclear):
    d = 0j
    dx = 0j
    dy = 0j
    dz = 0j
    k0 = m[q, 0] - m_lo[0]
    k1 = m[q, 1] - m_lo[1]
    k2 = m[q, 2] - m_lo[2]
    for t in range(n_terms):
        e = E[t, 0, k0] * E[t, 1, k1] * E[t, 2, k2]
        if nuclear:
            p = coef[t, 0] if tt[t, 0] < 0 else coef[t, 0] * T[tt[t, 0], q]
            if coef[t, 1] != 0:
                p += coef[t, 1] if tt[t, 1] < 0 else coef[t, 1] * T[tt[t, 1], q]
            d += p * e
        if mcoef[t, 0] != 0:
            a = mcoef[t, 0] if mtt[t, 0] < 0 else mcoef[t, 0] * Tm[mtt[t, 0], q]
            if mcoef[t, 1] != 0:
                a += mcoef[t, 1] if mtt[t, 1] < 0 else mcoef[t, 1] * Tm[mtt[t, 1], q]
            ae = a * e
            dx += ae * vec[t, 0]
            dy += ae * vec[t, 1]
            dz += ae * vec[t, 2]
    return d, dx, dy, dz


@njit(parallel=True, cache=True)
def evaluate_magnetic(m, B, F, FM, T, Tm, m_lo, E, n_terms, coef, tt, mcoef, mtt, vec, nuclear, inv_n, n_ops,
                      w, wy, offset, has_offset, kb, tb, has_basis, partial, chunk_sums):
    """Candidate observation sums of a move with magnetic scattering (see :func:`evaluate_nuclear`)."""
    n_obs = w.shape[0]
    n_chunks = partial.shape[0]
    K = partial.shape[1]
    inv_ops = 1.0 / n_ops
    for c in prange(n_chunks):
        if has_basis:
            for k in range(K):
                partial[c, k] = 0.0
        sx = 0.0
        sxx = 0.0
        sxy = 0.0
        for o in range(c * n_obs // n_chunks, (c + 1) * n_obs // n_chunks):
            v = 0.0
            for r in range(n_ops):
                q = o + r * n_obs
                d, dx, dy, dz = _delta_mag(m, q, m_lo, E, n_terms, coef, tt, T, mcoef, mtt, Tm, vec, nuclear)
                x = inv_n * _I(m, B, q, F[q] + d, FM[q, 0] + dx, FM[q, 1] + dy, FM[q, 2] + dz, True)
                if has_offset:
                    x -= offset[q]
                v += x
            v *= inv_ops
            wx = w[o] * v
            sx += wx
            sxx += wx * v
            sxy += wy[o] * v
            if has_basis:
                partial[c, kb[o]] += wx * (1.0 - tb[o])
                partial[c, kb[o] + 1] += wx * tb[o]
        chunk_sums[c, 0] = sx
        chunk_sums[c, 1] = sxx
        chunk_sums[c, 2] = sxy
    sx = 0.0
    sxx = 0.0
    sxy = 0.0
    for c in range(n_chunks):
        sx += chunk_sums[c, 0]
        sxx += chunk_sums[c, 1]
        sxy += chunk_sums[c, 2]
    return sx, sxx, sxy


@njit(parallel=True, cache=True)
def apply_magnetic(m, F, FM, T, Tm, m_lo, E, n_terms, coef, tt, mcoef, mtt, vec, nuclear):
    for q in prange(F.shape[0]):
        d, dx, dy, dz = _delta_mag(m, q, m_lo, E, n_terms, coef, tt, T, mcoef, mtt, Tm, vec, nuclear)
        F[q] += d
        FM[q, 0] += dx
        FM[q, 1] += dy
        FM[q, 2] += dz


@njit(cache=True)
def site_penalty(su, suu, n, Linv):
    r"""
    Deviation of a kind's displacement statistics from its ADP.

    In whitened coordinates :math:`\mathbf{z} = L^{-1}\mathbf{u}` with
    :math:`U = LL^T`,

    .. math::

        g = \lVert C_z - I \rVert_F^2 + \lVert \bar{\mathbf{z}} \rVert^2

    where :math:`C_z` and :math:`\bar{\mathbf{z}}` are the covariance and
    mean of the displacements of the atoms of one kind.

    Parameters
    ----------
    su : ndarray, (3,)
        Sum of displacements.
    suu : ndarray, (3, 3)
        Sum of outer products.
    n : int
        Number of atoms.
    Linv : ndarray, (3, 3)
        Inverse Cholesky factor of the ADP.

    Returns
    -------
    g : float
        Penalty.

    """

    mean = su / n
    C = suu / n - np.outer(mean, mean)
    Cz = Linv @ C @ Linv.T
    mz = Linv @ mean
    g = 0.0
    for a in range(3):
        g += mz[a] * mz[a]
        for b in range(3):
            d = Cz[a, b] - (1.0 if a == b else 0.0)
            g += d * d
    return g


@njit(cache=True)
def _penalty_change(n_change, k_old, k_new, u_old, u_new, sum_u, sum_uu, n_kind,
                    Linv, penalized, g, aff, aff_su, aff_suu, aff_g):
    """
    Candidate displacement sums and penalties of the kinds a move touches.

    Atom ``c`` leaves kind ``k_old[c]`` with displacement ``u_old[c]`` and
    joins ``k_new[c]`` with ``u_new[c]``. Affected penalized kinds are
    written to ``aff[:n_aff]`` with their candidate sums and penalties.

    Returns
    -------
    n_aff : int
        Number of affected kinds.
    dg : float
        Change of the summed penalty.

    """

    n_aff = 0
    for c in range(n_change):
        for side in range(2):
            k = k_old[c] if side == 0 else k_new[c]
            if penalized[k] == 0:
                continue
            a = -1
            for b in range(n_aff):
                if aff[b] == k:
                    a = b
            if a < 0:
                a = n_aff
                aff[a] = k
                aff_su[a] = sum_u[k]
                aff_suu[a] = sum_uu[k]
                n_aff += 1
            sign = -1.0 if side == 0 else 1.0
            for e in range(3):
                v = u_old[c, e] if side == 0 else u_new[c, e]
                aff_su[a, e] += sign * v
                for f in range(3):
                    w = u_old[c, f] if side == 0 else u_new[c, f]
                    aff_suu[a, e, f] += sign * v * w
    dg = 0.0
    for a in range(n_aff):
        k = aff[a]
        aff_g[a] = site_penalty(aff_su[a], aff_suu[a], n_kind[k], Linv[k])
        dg += aff_g[a] - g[k]
    return n_aff, dg


@njit(cache=True)
def _random_atom(sites, n_sites, n_cells, kind, penalized):
    """Random atom on one of ``sites`` whose kind is penalized (not vacant)."""
    for _ in range(100):
        j = sites[np.random.randint(sites.shape[0])]
        i = j + n_sites * np.random.randint(n_cells)
        if penalized[kind[i]]:
            return i
    return -1


@njit(cache=True)
def _exchange_pair(sites, n_sites, n_cells, kind, penalized):
    """
    Two atoms on the same one of ``sites`` with at least one penalized
    (occupied) kind, or ``(-1, -1)``.
    """
    for _ in range(100):
        j = sites[np.random.randint(sites.shape[0])]
        a = j + n_sites * np.random.randint(n_cells)
        b = j + n_sites * np.random.randint(n_cells)
        if a != b and (penalized[kind[a]] or penalized[kind[b]]):
            return a, b
    return -1, -1


@njit(cache=True)
def _cone(S, cos_max):
    """Random unit vector within angle ``acos(cos_max)`` of unit vector S."""
    ct = 1.0 - np.random.random() * (1.0 - cos_max)
    st = np.sqrt(max(0.0, 1.0 - ct * ct))
    phi = 2.0 * np.pi * np.random.random()
    if abs(S[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, S)
    e1 /= np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
    e2 = np.cross(S, e1)
    v = ct * S + st * (np.cos(phi) * e1 + np.sin(phi) * e2)
    return v / np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@njit(cache=True)
def run(
    n_moves,
    points,
    data,
    factors,
    F,
    FM,
    inv_n,
    totals,
    fit_mode,
    scale,
    background,
    config,
    moves,
    displacement,
    temperature,
    decay,
    out,
    basis,
    obs,
    fused,
):
    """
    Metropolis refinement for ``n_moves`` proposals.

    Parameters
    ----------
    totals : tuple
        ``(sw, sy, syy, dof)``.
    config : tuple
        ``(cell, site, su, u, kind, S, n_sites, n_cells)``: cells, sites,
        scaled fractional displacements ``2 pi u / N``, Cartesian
        displacements, kinds, spins.
    moves : tuple
        ``(move_cdf, swap_sites, spin_sites, displace_sites,
        swap_displace_sites, cos_max)``. Exchange moves use
        ``displace_sites``.
    displacement : tuple
        ``(L, Linv, A_inv, tpN, step, penalty, su, suu, g, n_kind,
        penalized)``, per kind: ADP Cholesky factors and inverses,
        displacement sums, penalties, atom counts, and whether the kind's
        displacements are refined and constrained.
    out : tuple
        ``(cost, chi_sq, temperature, scale, background, move, accepted,
        delta)`` output arrays of length ``n_moves``.
    basis : tuple
        Background basis (see module notes); the recorded background is 0
        when it is used.
    obs : tuple
        ``(n_ops, w, wy, xobs)``: number of symmetry-equivalent points per
        observation (blocks of the point arrays), observation weights,
        weights times observations, and a buffer for the averaged model.
    fused : tuple
        ``(enabled, m_lo, E, coef, tt, mcoef, mtt, vec, xyz, tpN, theta,
        partial, chunk_sums)`` for the fused kernels (used for every move
        when enabled): lowest index per axis, term tables (MAX_TERMS, 3, L),
        nuclear and magnetic term coefficients and table indices, term spin
        vectors, site positions, ``2 pi / N``, and work arrays.

    The configuration, structure factors, and displacement sums are updated
    in place. Displacements belong to positions, so a swap moves each
    atom's displacement into the statistics of its new kind. An exchange
    swaps the displacements of two atoms on the same site, so swaps and
    exchanges together conserve the displacements of each site as a set. The cost is
    ``chi^2 / dof + penalty * mean(g)`` over the penalized kinds; moves are
    accepted with probability ``exp(-delta cost / T)`` and ``T`` decays by
    ``exp(-decay)`` per move.

    Returns
    -------
    temperature : float
        Temperature after the last move.

    """

    sw, sy, syy, dof = totals
    cell, site, s_u, u, kind, S, n_sites, n_cells = config
    move_cdf, swap_sites, spin_sites, displace_sites, swap_displace_sites, cos_max = moves
    L, Linv, A_inv, tpN, step, penalty, sum_u, sum_uu, g, n_kind, penalized = displacement
    (
        out_cost,
        out_chi_sq,
        out_temperature,
        out_scale,
        out_background,
        out_move,
        out_accepted,
        out_delta,
    ) = out
    magnetic = factors[7]
    use_fused, m_lo, E, coef, tt, mcoef, mtt, vec, xyz, tpN_f, theta, f_partial, f_chunks = fused
    m_pts = points[0]
    T_nuc, T_mag, nuclear = factors[2], factors[5], factors[6]
    w_o, wy_o = obs[1], obs[2]
    offset, has_offset = data[2], data[3]
    n_terms = 0

    n_pen = 0
    g_total = 0.0
    for k in range(penalized.shape[0]):
        if penalized[k]:
            n_pen += 1
            g_total += g[k]
    pen_norm = penalty / n_pen if n_pen > 0 else 0.0

    n_max = max(points[3].shape[0], points[4].shape[0], points[5].shape[0])
    c_lat = np.zeros((2, 3, n_max), dtype=np.complex128)
    c_site = np.zeros(2, dtype=np.int64)
    c_su_old = np.zeros((2, 3))
    c_su_new = np.zeros((2, 3))
    k_old = np.zeros(2, dtype=np.int64)
    k_new = np.zeros(2, dtype=np.int64)
    u_old = np.zeros((2, 3))
    u_new = np.zeros((2, 3))
    S_old = np.zeros((2, 3))
    S_new = np.zeros((2, 3))
    dS = np.zeros(3)
    idx = np.zeros(2, dtype=np.int64)
    aff = np.zeros(4, dtype=np.int64)
    aff_su = np.zeros((4, 3))
    aff_suu = np.zeros((4, 3, 3))
    aff_g = np.zeros(4)

    I = np.empty(F.shape[0])
    intensity(points, F, FM, inv_n, magnetic, I)
    sx, sxx, sxy = sums(I, data)
    sx, sxx, sxy = _observed(sx, sxx, sxy, data, obs)
    chi_sq, scale, background = _fit(
        sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background, obs, basis
    )
    cost = chi_sq / dof + pen_norm * g_total

    for t in range(n_moves):
        temperature *= np.exp(-decay)

        p = np.random.random()
        move = 0
        while move < move_cdf.shape[0] - 1 and p > move_cdf[move]:
            move += 1

        n_change = 1
        possible = True

        if move == MOVE_SWAP or move == MOVE_SWAP_DISPLACE:
            sites = swap_sites if move == MOVE_SWAP else swap_displace_sites
            j = sites[np.random.randint(sites.shape[0])]
            a = j + n_sites * np.random.randint(n_cells)
            b = a
            for _ in range(100):
                b = j + n_sites * np.random.randint(n_cells)
                if kind[b] != kind[a]:
                    break
            if kind[b] == kind[a]:
                possible = False
            else:
                n_change = 2
                idx[0] = a
                idx[1] = b
                k_old[0] = kind[a]
                k_old[1] = kind[b]
                k_new[0] = kind[b]
                k_new[1] = kind[a]
        elif move == MOVE_SPIN:
            j = spin_sites[np.random.randint(spin_sites.shape[0])]
            i = j + n_sites * np.random.randint(n_cells)
            idx[0] = i
            k_old[0] = kind[i]
            k_new[0] = kind[i]
        elif move == MOVE_EXCHANGE:
            a, b = _exchange_pair(displace_sites, n_sites, n_cells, kind, penalized)
            if a < 0:
                possible = False
            else:
                n_change = 2
                idx[0] = a
                idx[1] = b
                k_old[0] = kind[a]
                k_old[1] = kind[b]
                k_new[0] = kind[a]
                k_new[1] = kind[b]
        else:
            i = _random_atom(displace_sites, n_sites, n_cells, kind, penalized)
            if i < 0:
                possible = False
            else:
                idx[0] = i
                k_old[0] = kind[i]
                k_new[0] = kind[i]

        dg = 0.0
        n_aff = 0
        if possible:
            for c in range(n_change):
                i = idx[c]
                c_site[c] = site[i]
                lattice_tables(points, cell[i], c_lat[c])
                for e in range(3):
                    c_su_old[c, e] = s_u[i, e]
                    c_su_new[c, e] = s_u[i, e]
                    u_old[c, e] = u[i, e]
                    u_new[c, e] = u[i, e]
                    S_old[c, e] = S[i, e]
                    S_new[c, e] = S[i, e]

            if move == MOVE_SPIN:
                v = _cone(S[idx[0]], cos_max)
                for e in range(3):
                    S_new[0, e] = v[e]
                    dS[e] = v[e] - S_old[0, e]
            elif move == MOVE_DISPLACE or move == MOVE_SWAP_DISPLACE:
                for c in range(n_change):
                    k = k_new[c]
                    if not penalized[k]:
                        continue
                    z = np.random.standard_normal(3)
                    du = step * (L[k] @ z)
                    du_frac = A_inv @ du
                    for e in range(3):
                        c_su_new[c, e] = c_su_old[c, e] + tpN[e] * du_frac[e]
                        u_new[c, e] = u_old[c, e] + du[e]
            elif move == MOVE_EXCHANGE:
                for e in range(3):
                    c_su_new[0, e] = c_su_old[1, e]
                    c_su_new[1, e] = c_su_old[0, e]
                    u_new[0, e] = u_old[1, e]
                    u_new[1, e] = u_old[0, e]

            if move != MOVE_SPIN:
                n_aff, dg = _penalty_change(
                    n_change, k_old, k_new, u_old, u_new, sum_u, sum_uu, n_kind,
                    Linv, penalized, g, aff, aff_su, aff_suu, aff_g,
                )

        ok = False
        d_cost = np.nan
        chi_cand, s_cand, b_cand = chi_sq, scale, background
        if possible and use_fused:
            n_terms = _move_terms(move, n_change, idx, cell, xyz, c_site, c_su_old, c_su_new, k_old, k_new,
                                  S_old, dS, factors, tpN_f, m_lo, E, coef, tt, mcoef, mtt, vec, theta)
            if magnetic:
                sx, sxx, sxy = evaluate_magnetic(
                    m_pts, points[2], F, FM, T_nuc, T_mag, m_lo, E, n_terms, coef, tt, mcoef, mtt, vec,
                    nuclear, inv_n, obs[0], w_o, wy_o, offset, has_offset, basis[1], basis[2], basis[0],
                    f_partial, f_chunks,
                )
            else:
                sx, sxx, sxy = evaluate_nuclear(
                    m_pts, F, T_nuc, m_lo, E, n_terms, coef, tt, inv_n, obs[0],
                    w_o, wy_o, offset, has_offset, basis[1], basis[2], basis[0], f_partial, f_chunks,
                )
            chi_cand, s_cand, b_cand = _fit_nuclear(
                sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background, basis, f_partial
            )
        elif possible:
            if move == MOVE_SWAP:
                sx, sxx, sxy = evaluate_swap(
                    points, data, data[4], factors, F, FM, inv_n, c_lat, c_site, c_su_old,
                    k_old[0], k_old[1], S_old[0], S_old[1],
                )
            elif move == MOVE_SPIN:
                sx, sxx, sxy = evaluate_spin(
                    points, data, data[4], factors, F, FM, inv_n, c_lat, c_site[0],
                    c_su_old[0], k_old[0], dS,
                )
            elif move == MOVE_DISPLACE:
                sx, sxx, sxy = evaluate_displace(
                    points, data, data[4], factors, F, FM, inv_n, c_lat, c_site[0],
                    c_su_old[0], c_su_new[0], k_old[0], S_old[0],
                )
            else:
                sx, sxx, sxy = evaluate_pair(
                    points, data, data[4], factors, F, FM, inv_n, c_lat, c_site, c_su_old,
                    c_su_new, k_old, k_new, S_old,
                )
            sx, sxx, sxy = _observed(sx, sxx, sxy, data, obs)
            chi_cand, s_cand, b_cand = _fit(
                sw, sy, syy, sx, sxx, sxy, fit_mode, scale, background, obs, basis
            )
        if possible:
            g_cand = g_total + dg
            cost_cand = chi_cand / dof + pen_norm * g_cand

            d_cost = cost_cand - cost
            ok = d_cost <= 0.0
            if not ok:
                ok = np.random.random() < np.exp(-d_cost / temperature)

            if ok:
                if use_fused and magnetic:
                    apply_magnetic(m_pts, F, FM, T_nuc, T_mag, m_lo, E, n_terms, coef, tt, mcoef, mtt, vec,
                                   nuclear)
                elif use_fused:
                    apply_nuclear(m_pts, F, T_nuc, m_lo, E, n_terms, coef, tt)
                elif move == MOVE_SWAP:
                    apply_swap(
                        points, factors, F, FM, c_lat, c_site, c_su_old,
                        k_old[0], k_old[1], S_old[0], S_old[1],
                    )
                elif move == MOVE_SPIN:
                    apply_spin(points, factors, FM, c_lat, c_site[0], c_su_old[0], k_old[0], dS)
                elif move == MOVE_DISPLACE:
                    apply_displace(
                        points, factors, F, FM, c_lat, c_site[0], c_su_old[0],
                        c_su_new[0], k_old[0], S_old[0],
                    )
                else:
                    apply_pair(
                        points, factors, F, FM, c_lat, c_site, c_su_old, c_su_new,
                        k_old, k_new, S_old,
                    )
                for c in range(n_change):
                    i = idx[c]
                    kind[i] = k_new[c]
                    for e in range(3):
                        s_u[i, e] = c_su_new[c, e]
                        u[i, e] = u_new[c, e]
                        S[i, e] = S_new[c, e]
                for a in range(n_aff):
                    k = aff[a]
                    sum_u[k] = aff_su[a]
                    sum_uu[k] = aff_suu[a]
                    g[k] = aff_g[a]
                g_total = g_cand
                cost = cost_cand
                chi_sq = chi_cand
                scale = s_cand
                background = b_cand

        out_move[t] = move
        out_accepted[t] = 1 if ok else 0
        out_delta[t] = d_cost
        out_cost[t] = cost
        out_chi_sq[t] = chi_sq
        out_temperature[t] = temperature
        out_scale[t] = scale
        out_background[t] = background

    return temperature
