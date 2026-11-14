r"""
Pair correlations of supercell configurations.

Atom :math:`(\mathbf{R}, j)` sits at :math:`\mathbf{R} + \mathbf{x}_j`. For
site fields :math:`a_j(\mathbf{R})` with presence masks
:math:`m_j(\mathbf{R})`, the correlation of sites :math:`j, j'` at lattice
separation :math:`\Delta\mathbf{R}` is

.. math::

    C_{jj'}(\Delta\mathbf{R}) =
        \frac{\sum_\mathbf{R} a_j(\mathbf{R}) \cdot a_{j'}(\mathbf{R} + \Delta\mathbf{R})}
             {\sum_\mathbf{R} m_j(\mathbf{R}) m_{j'}(\mathbf{R} + \Delta\mathbf{R})}

with separation vector :math:`\mathbf{r} = \Delta\mathbf{R} + \mathbf{x}_{j'} -
\mathbf{x}_j` (minimum image). The supercell is periodic, so both sums are
cyclic cross-correlations over the cell grid and every separation is
obtained at once by FFT, with no neighbor search or cutoff.

Symmetrization averages over the crystal's space-group operators. Each maps
pair :math:`(j, j', \Delta\mathbf{R})` to :math:`(\sigma j, \sigma j', W
\Delta\mathbf{R} + \mathbf{t}_{j'} - \mathbf{t}_j)`, where :math:`W
\mathbf{x}_j + \mathbf{w} = \mathbf{x}_{\sigma j} + \mathbf{t}_j`;
displacement tensors also rotate. Numerators and pair counts are summed over
operators before dividing.
"""

from dataclasses import dataclass

import numpy as np

QUANTITIES = ("occupancy", "spin", "displacement")


@dataclass
class Correlations:
    """
    Pair correlations of one quantity.

    Attributes
    ----------
    quantity : str
        ``"occupancy"``, ``"spin"``, or ``"displacement"``.
    sites : ndarray, (n, 2), int
        Active-site indices :math:`j, j'` of each pair.
    labels : list of str
        ``"site-site"``, each site named by its occupants' CIF labels
        (e.g. ``"Fe1/Mn1-Fe2/Mn2"``).
    orbits : ndarray, (n, 2), int
        Asymmetric-unit indices of the two sites.
    dR : ndarray, (n, 3), int
        Lattice separation (minimum image of the pair vector).
    r : ndarray, (n, 3)
        Cartesian separation vector (A).
    distance : ndarray, (n,)
        Pair distance (A).
    value : ndarray, (n,)
        Correlation: occupancy covariance :math:`\\langle\\delta_i
        \\delta_j\\rangle`, :math:`\\langle\\mathbf{S}_i \\cdot
        \\mathbf{S}_j\\rangle`, or :math:`\\langle\\mathbf{u}_i \\cdot
        \\mathbf{u}_j\\rangle` (A^2); normalized to a correlation
        coefficient if requested.
    error : ndarray, (n,)
        Standard error over configurations (zero for one configuration).
    count : ndarray, (n,)
        Number of contributing atom pairs per configuration.
    tensor : ndarray, (n, 3, 3) or None
        Cartesian :math:`\\langle\\mathbf{u}_i \\otimes \\mathbf{u}_j\\rangle`
        (A^2) for displacements.

    """

    quantity: str
    sites: np.ndarray
    labels: list
    orbits: np.ndarray
    dR: np.ndarray
    r: np.ndarray
    distance: np.ndarray
    value: np.ndarray
    error: np.ndarray
    count: np.ndarray
    tensor: np.ndarray = None

    def __len__(self):
        return len(self.value)

    def select(self, keep):
        """Subset of the pairs."""
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep
        return Correlations(
            self.quantity,
            self.sites[idx],
            [self.labels[i] for i in idx],
            self.orbits[idx],
            self.dR[idx],
            self.r[idx],
            self.distance[idx],
            self.value[idx],
            self.error[idx],
            self.count[idx],
            None if self.tensor is None else self.tensor[idx],
        )

    def shells(self, tol=1e-3, r_max=None):
        """
        Correlations averaged over pairs of equal distance and orbit pair.

        Parameters
        ----------
        tol : float, optional
            Distances within ``tol`` (A) form one shell.
        r_max : float, optional
            Largest distance.

        Returns
        -------
        shells : dict of ndarray
            ``label``, ``distance``, ``value``, ``error``, and ``n`` (number
            of pair vectors), sorted by orbit pair then distance. Self terms
            (zero distance) are excluded.

        """

        keep = self.distance > 1e-8
        if r_max is not None:
            keep &= self.distance <= r_max
        c = self.select(keep)
        orbit = np.sort(c.orbits, axis=1)
        d_key = np.round(c.distance / tol).astype(np.int64)
        keys = np.column_stack([orbit, d_key])
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        w = c.count.astype(float)
        sw = np.bincount(inverse, w)
        value = np.bincount(inverse, w * c.value) / sw
        error = np.sqrt(np.bincount(inverse, (w * c.error) ** 2)) / sw
        distance = np.bincount(inverse, w * c.distance) / sw
        n = np.bincount(inverse)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(inverse))[::-1]
        labels = np.array(["-".join(sorted(c.labels[i].split("-"))) for i in first])
        return {"label": labels, "distance": distance, "value": value, "error": error, "n": n}


def _grid(sc, x):
    """Per-atom array (n_atoms, d) as (n_sites, N0, N1, N2, d)."""
    d = x.shape[1]
    x = x.reshape(*sc.N, sc.n_sites, d)
    return np.moveaxis(x, 3, 0)


def _fields(sc, quantity, kind, u, S, kinds):
    """Fields, masks, and the active sites they live on."""

    n = len(sc)
    vacant = np.array([k.vacancy for k in sc.kinds])
    if quantity == "occupancy":
        channels, chosen = [], []
        for j, ks in enumerate(sc.site_kinds):
            if len(ks) < 2:
                continue
            if kinds is not None:
                k = [k for k in ks if sc.kinds[k].label in kinds]
                if not k:
                    continue
                k = k[0]
            else:
                k = next(k for k in ks if not vacant[k])
            channels.append(j)
            chosen.append(k)
        if not channels:
            raise ValueError("no mixed sites for occupancy correlations")
        sel = np.full(sc.n_sites, -1)
        sel[channels] = chosen
        k_atom = sel[sc.site]
        frac = np.array([sc.kinds[k].fraction for k in chosen])
        a = np.zeros((n, 1))
        on = k_atom >= 0
        a[on, 0] = (kind[on] == k_atom[on]) - np.array([sc.kinds[k].fraction for k in k_atom[on]])
        m = on[:, None].astype(float)
        names = [sc.kinds[k].label for k in chosen]
        return np.array(channels), a, m, (frac, names)
    if quantity == "spin":
        magnetic = np.array([k.magnetic for k in sc.kinds])
        channels = np.array([j for j, ks in enumerate(sc.site_kinds) if magnetic[ks].any()])
        if channels.size == 0:
            raise ValueError("no magnetic sites for spin correlations")
        mk = magnetic[kind].astype(float)
        return channels, S * mk[:, None], mk[:, None], None
    if quantity == "displacement":
        occupied = (~vacant[kind]).astype(float)
        return np.arange(sc.n_sites), u * occupied[:, None], occupied[:, None], None
    raise ValueError(f"unknown quantity {quantity!r}; choose from {QUANTITIES}")


def _cross(a, b, tensor=False):
    """Cyclic cross-correlation sums over the cell grid, per channel pair."""
    axes = (1, 2, 3)
    A = np.conj(np.fft.fftn(a, axes=axes))
    B = np.fft.fftn(b, axes=axes)
    if tensor:
        P = np.einsum("i...a,j...b->ij...ab", A, B)
        return np.fft.ifftn(P, axes=(2, 3, 4)).real
    P = np.einsum("i...d,j...d->ij...", A, B)
    return np.fft.ifftn(P, axes=(2, 3, 4)).real


def _site_maps(sc, channels):
    """Channel permutation and lattice shifts of each space-group operator."""

    crystal = sc.crystal
    xyz = np.array([sc.sites[j].xyz for j in channels])
    N = np.array(sc.N)
    A, A_inv = crystal.A, np.linalg.inv(crystal.A)
    maps = []
    for W, w in crystal.operators:
        M = np.diag(1 / N) @ W @ np.diag(N)
        if not np.allclose(M, np.round(M)):
            raise ValueError(f"supercell {sc.N} is not invariant under operator {W.tolist()}")
        sigma = np.empty(len(channels), dtype=np.int64)
        t = np.empty((len(channels), 3), dtype=np.int64)
        for c, x in enumerate(xyz):
            y = W @ x + w
            d = y - xyz
            hit = np.flatnonzero(np.all(np.abs(d - np.round(d)) < 1e-4, axis=1))
            if hit.size != 1:
                raise ValueError("active sites are not closed under the space group")
            sigma[c] = hit[0]
            t[c] = np.round(y - xyz[hit[0]]).astype(np.int64)
        maps.append((W, sigma, t, A @ W @ A_inv))
    return maps


def _symmetrize(sc, channels, num, den, tensor, labels_ok):
    N = np.array(sc.N)
    n_ch = len(channels)
    G = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    flat_num = num.reshape(n_ch, n_ch, -1, *num.shape[5:])
    flat_den = den.reshape(n_ch, n_ch, -1)
    out_num = np.zeros_like(flat_num)
    out_den = np.zeros_like(flat_den)
    maps = _site_maps(sc, channels)
    for W, sigma, t, Rc in maps:
        labels_ok(sigma)
        WG = G @ W.T
        for c in range(n_ch):
            for e in range(n_ch):
                R = np.mod(WG + t[e] - t[c], N)
                f = np.ravel_multi_index(R.T, N)
                v = flat_num[c, e]
                if tensor:
                    v = Rc @ v @ Rc.T
                out_num[sigma[c], sigma[e], f] += v
                out_den[sigma[c], sigma[e], f] += flat_den[c, e]
    k = len(maps)
    return out_num.reshape(num.shape) / k, out_den.reshape(den.shape) / k


def correlations(
    supercell,
    quantity,
    states=None,
    symmetrize=True,
    normalize=False,
    kinds=None,
):
    """
    Pair correlations of configurations of a supercell.

    Parameters
    ----------
    supercell : Supercell
        Geometry, kinds, and (if ``states`` is omitted) the configuration.
    quantity : {"occupancy", "spin", "displacement"}
        Occupancy deviations :math:`\\delta = [k_i = k] - c_k` on mixed sites,
        spins on atoms of magnetic kinds, or displacements of occupied atoms.
    states : iterable of dict, optional
        Configurations with keys ``kind``, ``u``, ``S`` (missing keys taken
        from ``supercell``), e.g. the saved states of independent runs.
    symmetrize : bool, optional
        Average over the crystal's space-group operators. The supercell must
        be invariant under them (e.g. equal N along symmetry-related axes).
    normalize : bool, optional
        Return correlation coefficients: occupancy covariances divided by
        :math:`\\sqrt{c_j(1-c_j) c_{j'}(1-c_{j'})}` (Warren-Cowley-like),
        displacement products by :math:`\\sqrt{\\langle u^2\\rangle_j
        \\langle u^2\\rangle_{j'}}`. Spins are unit vectors already.
    kinds : iterable of str, optional
        Occupant labels whose occupancy is correlated, one per mixed site;
        the first occupied kind of each site by default.

    Returns
    -------
    corr : Correlations
        Every site pair at every lattice separation, including self terms.

    """

    sc = supercell
    if states is None:
        states = [{}]
    tensor = quantity == "displacement"

    results = []
    for state in states:
        kind = np.asarray(state.get("kind", sc.kind))
        u = np.asarray(state.get("u", sc.u), dtype=float)
        S = np.asarray(state.get("S", sc.S), dtype=float)
        channels, a, m, extra = _fields(sc, quantity, kind, u, S, kinds)
        frac, names = extra if extra is not None else (None, None)
        ga = _grid(sc, a)[channels]
        gm = _grid(sc, m)[channels]
        num = _cross(ga, ga, tensor=tensor)
        den = _cross(gm, gm)
        if symmetrize:

            def labels_ok(sigma):
                if names is not None and any(names[c] != names[s] for c, s in enumerate(sigma)):
                    raise ValueError("symmetry maps an occupancy channel to a different occupant")

            num, den = _symmetrize(sc, channels, num, den, tensor, labels_ok)
        with np.errstate(invalid="ignore", divide="ignore"):
            if tensor:
                T = num / den[..., None, None]
                value = np.trace(T, axis1=-2, axis2=-1)
            else:
                T = None
                value = num / den
        if normalize:
            if quantity == "occupancy":
                v = frac * (1 - frac)
                value = value / np.sqrt(v[:, None] * v[None, :])[:, :, None, None, None]
            elif quantity == "displacement":
                self_term = np.array([value[c, c, 0, 0, 0] for c in range(len(channels))])
                value = value / np.sqrt(self_term[:, None] * self_term[None, :])[:, :, None, None, None]
        results.append((value, T, den))

    values = np.array([r[0] for r in results])
    value = values.mean(axis=0)
    error = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(value)
    T = None if not tensor else np.mean([r[1] for r in results], axis=0)
    den = np.mean([r[2] for r in results], axis=0)

    return _table(sc, quantity, channels, value, error, den, T)


def _table(sc, quantity, channels, value, error, den, T):
    N = np.array(sc.N)
    n_ch = len(channels)
    G = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    xyz = np.array([sc.sites[j].xyz for j in channels])
    c, e, g = np.meshgrid(np.arange(n_ch), np.arange(n_ch), np.arange(len(G)), indexing="ij")
    c, e, g = c.ravel(), e.ravel(), g.ravel()
    dx = xyz[e] - xyz[c]
    frac = G[g] + dx
    dR = G[g] - N * np.round(frac / N).astype(np.int64)
    r = sc.crystal.cartesian(dR + dx)
    count = den.reshape(-1)
    keep = count > 0
    orbit = np.array([sc.sites[j].orbit for j in channels])
    names = ["/".join(sc.sites[j].labels) for j in channels]
    idx = np.flatnonzero(keep)
    return Correlations(
        quantity=quantity,
        sites=np.column_stack([np.asarray(channels)[c[idx]], np.asarray(channels)[e[idx]]]),
        labels=[f"{names[c[i]]}-{names[e[i]]}" for i in idx],
        orbits=np.column_stack([orbit[c[idx]], orbit[e[idx]]]),
        dR=dR[idx],
        r=r[idx],
        distance=np.linalg.norm(r[idx], axis=1),
        value=value.reshape(-1)[idx],
        error=error.reshape(-1)[idx],
        count=count[idx],
        tensor=None if T is None else T.reshape(-1, 3, 3)[idx],
    )
