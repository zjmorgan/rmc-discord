r"""
Fast forward model: refined correlations to intensities by NUFFT.

The refined model is a sum over pair vectors,

.. math::

    I(\mathbf{h}) = \frac{s}{n_W} \sum_W \sum_g \sum_\alpha
    w_{g\alpha}(\mathbf{h}W)
    \sum_{p \in g} a_{p\alpha}\, \beta(W\mathbf{d}_p)\,
    e^{2\pi i\, \mathbf{h} \cdot W\mathbf{d}_p},

where the pairs fall into a few groups :math:`g` sharing the
:math:`\mathbf{Q}`-dependent prefactor (Debye-Waller factors of the two
sites, magnetic amplitudes), :math:`\alpha` runs over the components of
a channel (1 for occupancy and isotropic spins, 3 for the size effect,
6 for displacement and spin tensors) with weights
:math:`w_{g\alpha}` (e.g. :math:`Q_\alpha\, e^{-W_i - W_j}`), and the
bin factor :math:`\beta` is a window on the pair vectors. The inner sum
is a Fourier transform from scattered pair vectors to reciprocal space:

* on a whole volume, :math:`\mathbf{h} = W_\mathrm{v}\mathbf{u}` with
  :math:`\mathbf{u}` on a regular grid, so it is a type-1 NUFFT that
  returns every bin at once;
* at arbitrary positions it is a type-3 NUFFT.

Either costs about :math:`O(N_\mathrm{pairs} + M \log M)` per group and
component, instead of :math:`O(N_\mathrm{pairs} M)` for the design
matrix, and gives the same intensities to the NUFFT tolerance.
"""

import numpy as np

from rmc_discord.pairs.design import _quadratic_components, _tensor_components, bin_factor
from rmc_discord.pairs.symmetry import coefficient_slots

_DIAG = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _channel_terms(block, c):
    """
    Per-pair amplitudes of one channel.

    Returns
    -------
    coef : ndarray, (n_pairs, n_comp)
        Pair amplitudes per component (scattering lengths folded in for
        nuclear channels).
    weight : callable
        ``weight(Q) -> (m, n_comp)`` component weights at Cartesian ``Q``.
    trig : {"cos", "-sin"}
    """

    o, sites = block.orbits, block.sites
    magnetic = block.name == "spin"
    bb = np.ones(len(o)) if magnetic else sites.b[o.i] * sites.b[o.j]
    if block.kind == "orbit":
        cc = np.zeros(o.n_orbits)
        cc[block.columns] = c
        scale = 2.0 / 3.0 if magnetic else 1.0
        coef = (scale * bb * cc[o.orbit])[:, None]
        return coef, lambda Q: np.ones((len(Q), 1)), "cos"
    shape = (3,) if block.kind == "vector" else (3, 3)
    index, value = coefficient_slots(block.fields, len(o), shape)
    used = index >= 0
    coeff = np.where(used, c[np.maximum(index, 0)], 0.0)
    total = np.einsum("ps,ps...->p...", coeff, value)  # sigma_p (3,) or V_p (3, 3)
    if block.kind == "vector":  # size effect: -sum b b (Q . sigma) sin
        return bb[:, None] * total, lambda Q: Q, "-sin"
    V6 = _tensor_components(total) * bb[:, None]
    if magnetic:  # tr V - Qhat^T V Qhat

        def weight(Q):
            n = np.linalg.norm(Q, axis=1, keepdims=True)
            return _DIAG[None, :] - _quadratic_components(Q / np.where(n > 0, n, 1.0))

        return V6, weight, "cos"
    return V6, _quadratic_components, "cos"


def _groups(block, thermal):
    """
    Pairs sharing the Q-dependent prefactor.

    The prefactor of a pair is a product of per-site factors,
    :math:`e^{-W_i} e^{-W_j}` times :math:`a_i a_j` (or :math:`a^2_i` on
    magnetic self pairs), so pairs are grouped by the (unordered) classes
    of their two sites.

    Returns
    -------
    labels : ndarray of int, (n_pairs,)
        Group of each pair.
    groups : ndarray of int, (n_groups, 5)
        ``(amp_i, amp_j, dw_i, dw_j, self)`` class indices per group.
    amp_site, dw_site : ndarray of int, (n_sites,)
        Amplitude and Debye-Waller class of each site.
    """

    o, sites = block.orbits, block.sites
    n = len(sites)
    U = np.round(sites.U.reshape(n, 9), 10) if thermal else np.zeros((n, 9))
    _, dw_site = np.unique(U, axis=0, return_inverse=True)
    dw_site = dw_site.ravel()
    if block.amplitude is not None:
        a, a2 = block.amplitude(np.linspace(0.0, 15.0, 7))
        _, amp_site = np.unique(np.round(np.vstack([a, a2]).T, 10), axis=0, return_inverse=True)
        amp_site = amp_site.ravel()
        self_pair = (o.i == o.j) & np.all(o.R == 0, axis=1)
    else:
        amp_site = np.zeros(n, int)
        self_pair = np.zeros(len(o), bool)
    ki = np.column_stack([amp_site[o.i], dw_site[o.i]])
    kj = np.column_stack([amp_site[o.j], dw_site[o.j]])
    lo = (ki[:, 0] < kj[:, 0]) | ((ki[:, 0] == kj[:, 0]) & (ki[:, 1] <= kj[:, 1]))
    first, second = np.where(lo[:, None], ki, kj), np.where(lo[:, None], kj, ki)
    key = np.column_stack([first[:, 0], second[:, 0], first[:, 1], second[:, 1], self_pair])
    groups, labels = np.unique(key, axis=0, return_inverse=True)
    return labels.ravel(), groups, amp_site, dw_site


def _transforms(d, coef, targets, eps):
    """
    ``F[alpha] = sum_p coef[p, alpha] exp(2 pi i h . d_p)`` at the targets.

    ``targets`` is ``("grid", u0, du, shape, W)`` for a regular grid with
    ``h = W u`` or ``("points", h)``.
    """

    import finufft

    coef = np.ascontiguousarray(np.asarray(coef, complex).T)  # (n_comp, n_pairs): one batched call
    if targets[0] == "grid":
        _, u0, du, shape, Wv = targets
        q = d @ Wv  # h . d = u . (W^T d)
        off = np.array(shape) // 2
        x = 2 * np.pi * du * q
        x = np.mod(x + np.pi, 2 * np.pi) - np.pi
        c = np.ascontiguousarray(coef * np.exp(2j * np.pi * q @ (u0 + off * du)))
        xs = [np.ascontiguousarray(x[:, a]) for a in range(3)]
        F = finufft.nufft3d1(*xs, c, tuple(int(n) for n in shape), eps=eps, isign=1,
                             nthreads=_threads(np.prod(shape) + len(d)), upsampfac=_upsampling(eps))
        return F.reshape(len(coef), *shape)
    h = targets[1]
    xs = [np.ascontiguousarray(2 * np.pi * d[:, a]) for a in range(3)]
    ss = [np.ascontiguousarray(h[:, a]) for a in range(3)]
    F = finufft.nufft3d3(*xs, coef, *ss, eps=eps, isign=1, nthreads=_threads(len(h) + len(d)),
                         upsampfac=_upsampling(eps))
    return F.reshape(len(coef), len(h))


def _upsampling(eps):
    """Smaller fine grid (4x fewer FFT points) when the tolerance allows."""
    return 1.25 if eps >= 1e-8 else 2.0


def _threads(work):
    """One thread for small transforms (thread start-up dominates), all otherwise."""
    return 1 if work < 200_000 else 0


def forward(model, result, volume=None, h=None, widths=None, projection=None, channels=None,
            background=True, eps=1e-7):
    """
    Model intensity of a fit, by NUFFT.

    Parameters
    ----------
    model : PairModel
    result : FitResult
    volume : Volume, optional
        Evaluate on every bin of this volume (bin widths and projection
        taken from it).
    h : ndarray, (m, 3), optional
        Or at these Miller indices.
    widths, projection : optional
        Bin widths and projection for ``h`` (as for the design matrices).
    channels : sequence of str, optional
        Channels included (default all).
    background : bool, optional
        Add the refined background.
    eps : float, optional
        Relative NUFFT tolerance; the default is far below counting noise
        (``1e-10`` reproduces the design matrices to rounding).

    Returns
    -------
    I : ndarray
        ``volume.shape`` or ``(m,)``.

    """

    if volume is not None:
        widths, projection = np.asarray(volume.widths, float), np.asarray(volume.W, float)
        shape = tuple(volume.shape)
        u0 = np.array([e[0] for e in volume.edges]) + 0.5 * widths
        targets = ("grid", u0, widths, shape, projection)
        h = volume.hkl(np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3))
    else:
        h = np.asarray(h, float).reshape(-1, 3)
        targets = ("points", h)
        shape = (len(h),)
    out = np.zeros(len(h))

    operators = model.operators if model.operators is not None and len(model.operators) else [np.eye(3)]
    thermal = model.spec.thermal
    s = result.refinement.scale
    for block, start in zip(model.blocks, result.offsets):
        if channels is not None and block.name not in channels:
            continue
        c = result.refinement.c[start : start + block.n]
        coef, weight, trig = _channel_terms(block, c)
        labels, groups, amp_site, dw_site = _groups(block, thermal)
        live = np.any(coef != 0, axis=1)
        U6 = _tensor_components(block.sites.U)
        dw_rep = [np.flatnonzero(dw_site == k)[0] for k in range(dw_site.max() + 1)]
        amp_rep = [np.flatnonzero(amp_site == k)[0] for k in range(amp_site.max() + 1)]
        for W in operators:
            W = np.asarray(W, float)
            Q = model.structure.Q(h @ W)
            wq = weight(Q)  # (m, n_comp), the same for every group
            Q6 = _quadratic_components(Q)
            dw = [np.exp(-0.5 * Q6 @ U6[k]) if thermal else None for k in dw_rep]  # per site class
            del Q6
            if block.amplitude is not None:
                a, a2 = block.amplitude(np.linalg.norm(Q, axis=1), columns=amp_rep)
            del Q
            d = block.orbits.d @ W.T  # (h W) . d = h . (W d)
            beta = bin_factor(d, widths, projection) if widths is not None else np.ones(len(d))
            for g, (ai, aj, ui, uj, self_pair) in enumerate(groups):
                sel = (labels == g) & live
                if not sel.any():
                    continue
                F = _transforms(d[sel], coef[sel] * beta[sel, None], targets, eps).reshape(coef.shape[1], -1)
                part = F.real if trig == "cos" else -F.imag
                acc = np.einsum("am,ma->m", part, wq)
                if thermal:
                    acc *= dw[ui] * dw[uj]
                if block.amplitude is not None:
                    acc *= a2[:, ai] if self_pair else a[:, ai] * a[:, aj]
                out += acc * (s / len(operators))
    if background and result.background is not None and len(result.refinement.background):
        for k in range(0, len(h), 500_000):
            out[k : k + 500_000] += result.background(h[k : k + 500_000]) @ result.refinement.background
    return out.reshape(shape)
