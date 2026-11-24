r"""
Streaming least squares from accumulated normal equations.

Design matrices for :math:`10^6` intensities and hundreds of columns do
not fit in memory. They are row-separable, so the weighted Gram matrix
:math:`G = X^\mathsf{T} W X`, :math:`\mathbf{g} = X^\mathsf{T} W
\mathbf{y}` and :math:`y^\mathsf{T} W y` are accumulated tile by tile,
with :math:`X = [A \mid B]` the design columns of the pair orbits and the
background basis. The refinement of :func:`~rmc_discord.pairs.refine.refine`
(constraints :math:`\mathbf{c} = \mathbf{c}_0 + N\boldsymbol{\theta}`,
scale set by fixed coefficients, ridge, rank cutoff) then needs only
these :math:`k \times k` matrices, and the :math:`\chi^2` of any solution
on held-out data follows from that set's own normal equations.
"""

from dataclasses import dataclass, field

import numpy as np

from rmc_discord.pairs.refine import Refinement


@dataclass
class NormalEquations:
    """
    Weighted normal equations of columns ``[A | B]``.

    Attributes
    ----------
    n_design, n_background : int
        Number of design (orbit) and background columns.
    G : ndarray, (k, k)
        :math:`X^\\mathsf{T} W X`.
    g : ndarray, (k,)
        :math:`X^\\mathsf{T} W \\mathbf{y}`.
    yy : float
        :math:`\\mathbf{y}^\\mathsf{T} W \\mathbf{y}`.
    n_obs : int
        Rows added.

    """

    n_design: int
    n_background: int = 0
    G: np.ndarray = field(default=None)
    g: np.ndarray = field(default=None)
    yy: float = 0.0
    n_obs: int = 0

    def __post_init__(self):
        k = self.n_design + self.n_background
        if self.G is None:
            self.G = np.zeros((k, k))
        if self.g is None:
            self.g = np.zeros(k)

    def add(self, A, y, sigma, B=None):
        """Accumulate rows with design ``A``, background ``B``, data ``y`` and errors ``sigma``."""

        A = np.asarray(A, float)
        X = A if B is None or self.n_background == 0 else np.hstack([A, np.asarray(B, float)])
        if X.shape[1] != len(self.g):
            raise ValueError(f"expected {len(self.g)} columns, got {X.shape[1]}")
        w = 1.0 / np.asarray(sigma, float) ** 2
        Xw = X * w[:, None]
        self.G += X.T @ Xw
        self.g += Xw.T @ y
        self.yy += float(np.sum(w * y * y))
        self.n_obs += len(y)
        return self

    def chi2(self, q):
        """Weighted squared residual of column coefficients ``q`` (design then background)."""

        q = np.asarray(q, float)
        return float(self.yy - 2 * q @ self.g + q @ self.G @ q)

    def select(self, design=None, background=True):
        """Equations restricted to some design columns (and optionally no background)."""

        nd, nb = self.n_design, self.n_background
        design = np.arange(nd) if design is None else np.asarray(design)
        cols = np.concatenate([design, nd + np.arange(nb)]) if background else design
        return NormalEquations(len(design), nb if background else 0, self.G[np.ix_(cols, cols)].copy(),
                               self.g[cols].copy(), self.yy, self.n_obs)


def accumulate(design, h, y, sigma, background=None, split=None, tile=20000):
    """
    Normal equations for rows of ``h`` in tiles.

    Parameters
    ----------
    design : callable
        ``design(h_tile) -> A_tile`` (e.g. a closure over
        :func:`~rmc_discord.pairs.design.design_matrix`); several channels
        are concatenated by the caller.
    h, y, sigma : ndarray
        Positions (m, 3), intensities and uncertainties.
    background : callable, optional
        ``background(h_tile) -> B_tile``.
    split : ndarray of bool, optional
        Held-out rows; their equations are accumulated separately.
    tile : int, optional
        Rows per tile.

    Returns
    -------
    train : NormalEquations
    test : NormalEquations or None
        Held-out equations if ``split`` is given.

    """

    train = test = None
    for start in range(0, len(h), tile):
        sl = slice(start, start + tile)
        A = design(h[sl])
        B = background(h[sl]) if background is not None else None
        if train is None:
            nb = 0 if B is None else B.shape[1]
            train = NormalEquations(A.shape[1], nb)
            test = NormalEquations(A.shape[1], nb) if split is not None else None
        held = np.zeros(len(A), bool) if split is None else np.asarray(split[sl])
        if np.any(~held):
            train.add(A[~held], y[sl][~held], sigma[sl][~held], None if B is None else B[~held])
        if np.any(held):
            test.add(A[held], y[sl][held], sigma[sl][held], None if B is None else B[held])
    return train, test


def _transform(neq, parameterization):
    """Map parameters ``p = (s?, u, beta)`` to columns: ``q = M p``."""

    c0, N = parameterization.c0, parameterization.N
    nd, nb = neq.n_design, neq.n_background
    k = N.shape[1]
    has_scale = bool(np.any(c0 != 0))
    offset = 1 if has_scale else 0
    M = np.zeros((nd + nb, offset + k + nb))
    if has_scale:
        M[:nd, 0] = c0
    M[:nd, offset : offset + k] = N
    M[nd:, offset + k :] = np.eye(nb)
    return M, has_scale, offset, k


def refine_normal(neq, fixed=None, parameterization=None, ridge=0.0, rcond=1e-10, relative=False):
    r"""
    Refine from normal equations, as :func:`~rmc_discord.pairs.refine.refine`.

    The weighted Gram matrix of the parameters is diagonalized; its
    eigenvalues are the squared singular values of the weighted design
    matrix, cut at ``max(rcond, 1e-7)`` relative to the largest singular
    value (the normal equations square the condition number).

    Parameters
    ----------
    neq : NormalEquations
    fixed, parameterization, ridge, rcond
        As for :func:`~rmc_discord.pairs.refine.refine`.
    relative : bool, optional
        ``ridge`` is relative to the mean diagonal of the free-parameter
        normal matrix (independent of the data scale), e.g. ``1e-6`` to
        damp nearly collinear orbits.

    Returns
    -------
    result : Refinement
        ``chi2`` is the training value; ``predict`` works as usual.

    """

    from rmc_discord.pairs.constraints import parameterize

    if parameterization is None:
        parameterization = parameterize(neq.n_design, fixed)
    M, has_scale, offset, k = _transform(neq, parameterization)
    G = M.T @ neq.G @ M
    g = M.T @ neq.g
    if ridge > 0 and k:
        if relative:
            ridge = ridge * float(np.mean(np.diag(G)[offset : offset + k]))
        G[offset : offset + k, offset : offset + k] += ridge * np.eye(k)

    lam, V = np.linalg.eigh(G)
    lam_max = max(lam.max(), 0.0)
    keep = lam > (max(rcond, 1e-7) ** 2) * lam_max
    p = V[:, keep] @ ((V[:, keep].T @ g) / lam[keep])
    cov = (V[:, keep] / lam[keep]) @ V[:, keep].T

    q = M @ p
    chi2 = neq.chi2(q)  # data misfit only (ridge rows are not data)

    c0, N = parameterization.c0, parameterization.N
    s = p[0] if has_scale else 1.0
    u = p[offset : offset + k]
    c = c0 + N @ u / s
    J = N / s
    if has_scale:
        J = np.hstack([(-N @ u / s**2)[:, None], J])
        block = cov[: offset + k, : offset + k]
    else:
        block = cov[offset : offset + k, offset : offset + k]
    esd = np.sqrt(np.maximum(np.einsum("na,ab,nb->n", J, block, J), 0))
    cov_c = J @ block @ J.T if len(J) <= 8000 else None
    free = np.any(np.abs(N) > 1e-10, axis=1)
    beta = p[offset + k :]
    beta_esd = np.sqrt(np.maximum(np.diag(cov)[offset + k :], 0))
    sv = np.sqrt(np.clip(lam[::-1], 0, None))
    return Refinement(c, esd, free, float(s), float(np.sqrt(max(cov[0, 0], 0))) if has_scale else 0.0,
                      beta, beta_esd, chi2, neq.n_obs, M.shape[1], int(keep.sum()), sv, cov_c)


def column_coefficients(result):
    """Coefficients of the ``[A | B]`` columns: ``(s c, beta)`` concatenated."""

    return np.concatenate([result.scale * result.c, result.background])


def held_out_chi2(result, test):
    """Reduced chi^2 of a refinement on held-out normal equations."""

    return test.chi2(column_coefficients(result)) / max(1, test.n_obs)
