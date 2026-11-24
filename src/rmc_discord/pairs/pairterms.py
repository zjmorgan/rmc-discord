r"""Weighted Gaussian pair terms: the common currency of all models.

Every nuclear diffuse model is reduced to terms
:math:`(\mathbf{d}_p, w_p, U_p)` with

.. math::

    I(\mathbf{Q}) = \sum_p w_p \, e^{i \mathbf{Q} \cdot \mathbf{d}_p}
    \exp\left(-\tfrac{1}{2} \mathbf{Q}^\mathsf{T} U_p \mathbf{Q}\right),

equivalently a 3D-:math:`\Delta`\ PDF made of Gaussians of weight
:math:`w_p` and covariance :math:`U_p` centered on pair vectors
:math:`\mathbf{d}_p`. Occupational disorder on split sites gives
:math:`w = b_i b_j C_{ij}(\mathbf{R})` at
:math:`\mathbf{d} = \mathbf{R} + \mathbf{x}_j - \mathbf{x}_i` with
:math:`U = U_i + U_j` for uncorrelated thermal motion. Term sets must
be closed under :math:`(\mathbf{d}, w, U) \to (-\mathbf{d}, w, U)`,
which holds for any covariance with
:math:`C_{ij}(\mathbf{R}) = C_{ji}(-\mathbf{R})`; the intensity is
then real.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PairTerms:
    r"""
    Pair terms :math:`(\mathbf{d}, w, U)`.

    Attributes
    ----------
    d : ndarray, (n, 3)
        Pair vectors in fractional coordinates.
    w : ndarray, (n,)
        Weights in :math:`\mathrm{fm}^2` per unit cell.
    U : ndarray, (n, 3, 3) or None
        Cartesian Gaussian covariance in
        :math:`\mathring{\mathrm{A}}^2`, or ``None`` for point
        (unsmeared) terms.

    """

    d: np.ndarray
    w: np.ndarray
    U: np.ndarray = None

    def __len__(self):
        return len(self.w)

    @classmethod
    def concatenate(cls, terms):
        """Join term sets; missing covariances are taken as zero."""

        smeared = any(t.U is not None for t in terms)
        U = None
        if smeared:
            U = np.concatenate(
                [
                    t.U if t.U is not None else np.zeros((len(t), 3, 3))
                    for t in terms
                ]
            )
        d = np.concatenate([t.d for t in terms])
        w = np.concatenate([t.w for t in terms])
        return cls(d, w, U)

    def prune(self, tol):
        r"""Drop terms with :math:`|w| \le` ``tol``."""

        keep = np.abs(self.w) > tol
        U = self.U[keep] if self.U is not None else None
        return PairTerms(self.d[keep], self.w[keep], U)

    def bin_average(self, widths, projection=None):
        r"""
        Average over the histogram bins of a measured volume.

        A bin spans widths :math:`\Delta u_a` along the volume axes, with
        Miller indices :math:`\mathbf{h} = W\mathbf{u}`. Averaging
        :math:`e^{2\pi i \mathbf{h} \cdot \mathbf{d}}` over it multiplies
        each term by

        .. math::

            \prod_a \operatorname{sinc}\!\left(\Delta u_a\,
            (W^\mathsf{T}\mathbf{d})_a\right), \qquad
            \operatorname{sinc}(x) = \frac{\sin \pi x}{\pi x}.

        This is exact for point terms and neglects the variation of the
        Gaussian factor across a bin otherwise.

        Parameters
        ----------
        widths : array_like, (3,)
            Bin widths along the volume axes.
        projection : ndarray, (3, 3), optional
            Projection matrix :math:`W` (``Volume.W``); identity, i.e.
            axes along h, k, l, by default.

        Returns
        -------
        terms : PairTerms

        """

        from rmc_discord.pairs.design import bin_factor

        factor = bin_factor(self.d, widths, projection)
        return PairTerms(self.d, self.w * factor, self.U)

    def merge(self, decimals=8):
        r"""
        Combine reversed and coincident terms.

        Each :math:`(\mathbf{d}, U)` is mapped to a canonical sign of
        :math:`\mathbf{d}` and terms
        with identical keys are summed. Valid only for reversal-closed
        sets, for which the intensity is a cosine sum.

        Parameters
        ----------
        decimals : int, optional
            Rounding used to identify identical vectors and tensors.

        Returns
        -------
        terms : PairTerms
            Fewer terms with the same cosine-sum intensity.

        """

        d = self.d.copy()
        nonzero = np.abs(d) > 10 ** (-decimals)
        first = np.argmax(nonzero, axis=1)
        sign = np.sign(d[np.arange(len(d)), first])
        sign[sign == 0] = 1
        d *= sign[:, None]

        key = np.round(d, decimals)
        if self.U is not None:
            iu = np.triu_indices(3)
            key = np.hstack([key, np.round(self.U[:, iu[0], iu[1]], decimals)])
        key, index, inverse = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        w = np.bincount(inverse.ravel(), weights=self.w)
        U = self.U[index] if self.U is not None else None
        return PairTerms(d[index], w, U)


def pair_terms_from_covariance(sites, R, C, thermal=True, b=None):
    r"""
    Pair terms of occupational (split-site) covariances.

    Parameters
    ----------
    sites : Sites
        Participating positions.
    R : ndarray of int, (m, 3)
        Cell translations.
    C : ndarray, (m, n, n)
        Occupation covariance ``C[k, i, j]``
        :math:`= \langle \delta n_i(\mathbf{0})\,
        \delta n_j(\mathbf{R}_k) \rangle`.
    thermal : bool, optional
        Smear each term by :math:`U_i + U_j` (uncorrelated Gaussian
        motion, i.e. Debye-Waller factors :math:`e^{-W_i - W_j}`).
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.

    Returns
    -------
    terms : PairTerms

    """

    b = sites.b if b is None else np.asarray(b)
    n = len(sites)
    d = R[:, None, None, :] + sites.x[None, None, :] - sites.x[None, :, None]
    w = b[:, None] * b[None, :] * C
    U = None
    if thermal:
        U = sites.U[:, None] + sites.U[None, :]
        U = np.broadcast_to(U, (len(R), n, n, 3, 3)).reshape(-1, 3, 3)
    return PairTerms(d.reshape(-1, 3), w.ravel(), U)


def thermal_laue_terms(sites, b=None):
    r"""
    Incoherent thermal Laue terms of uncorrelated Gaussian motion.

    Adds :math:`x_i b_i^2 (1 - e^{-2 W_i})` per position, the self-term
    remainder of independent displacements. Correlated thermal diffuse
    scattering (phonons) is not described.

    Parameters
    ----------
    sites : Sites
        Positions with occupancies and displacement tensors.
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.

    Returns
    -------
    terms : PairTerms

    """

    b = sites.b if b is None else np.asarray(b)
    n = len(sites)
    w = sites.occupancy * b**2
    d = np.zeros((2 * n, 3))
    U = np.concatenate([np.zeros((n, 3, 3)), 2 * sites.U])
    return PairTerms(d, np.concatenate([w, -w]), U)
