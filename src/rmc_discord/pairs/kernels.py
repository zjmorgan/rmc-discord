"""
Fused, threaded evaluation of design rows.

Every channel's design matrix has the form

.. math::

    A_{rk} = \\sum_p \\sum_{s:\\,k_{ps}=k} t(2\\pi\\,\\mathbf{h}_r\\cdot\\mathbf{d}_p)\\,
    w_p\\, f_{r i_p} f_{r j_p}\\, \\boldsymbol{\\phi}_r\\cdot\\mathbf{c}_{ps},

with :math:`t` a cosine (or minus sine), pair weights :math:`w_p`
(scattering lengths and bin factor), per-row site factors :math:`f`
(Debye-Waller and magnetic amplitude), per-row features
:math:`\\boldsymbol{\\phi}_r` (1, :math:`\\mathbf{Q}`, the quadratic
components of :math:`\\mathbf{Q}`, ...) and per-pair coefficient values
:math:`\\mathbf{c}_{ps}` scattered into columns :math:`k_{ps}`. Evaluating
it in one pass over rows avoids the (rows x pairs) temporaries of the
NumPy reference, which dominate its cost.

"""

import numpy as np
from numba import njit, prange

TWO_PI = 2.0 * np.pi


@njit(parallel=True, cache=True)
def accumulate_rows(hw, d, w, si, sj, selfp, f, fself, sine, index, coef, phi, out):
    """
    Add design rows into ``out``.

    Parameters
    ----------
    hw : (t, 3) Miller indices (after the operator).
    d : (n, 3) pair vectors.
    w : (n,) pair weights.
    si, sj : (n,) site indices into the columns of ``f``.
    selfp : (n,) bool, pairs taking ``fself[:, si]`` instead of
        ``f[:, si] * f[:, sj]``.
    f, fself : (t, n_sites) per-row site factors.
    sine : bool, use ``-sin`` instead of ``cos``.
    index : (n, s) int, columns (``-1`` padding at the end).
    coef : (n, s, m) per-pair coefficient values.
    phi : (t, m) per-row features.
    out : (t, n_columns), accumulated in place.

    """

    t = hw.shape[0]
    n = d.shape[0]
    S = index.shape[1]
    m = phi.shape[1]
    for r in prange(t):
        h0 = TWO_PI * hw[r, 0]
        h1 = TWO_PI * hw[r, 1]
        h2 = TWO_PI * hw[r, 2]
        for p in range(n):
            x = h0 * d[p, 0] + h1 * d[p, 1] + h2 * d[p, 2]
            v = -np.sin(x) if sine else np.cos(x)
            if selfp[p]:
                v *= w[p] * fself[r, si[p]]
            else:
                v *= w[p] * f[r, si[p]] * f[r, sj[p]]
            for s in range(S):
                k = index[p, s]
                if k < 0:
                    break
                acc = 0.0
                for c in range(m):
                    acc += phi[r, c] * coef[p, s, c]
                out[r, k] += v * acc
