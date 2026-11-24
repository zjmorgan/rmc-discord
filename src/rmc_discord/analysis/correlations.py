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
        ``"occupancy"``, ``"spin"``, ``"displacement"``, or ``"bond"``
        (:func:`bond_lengths`).
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
        coefficient if requested. For ``"bond"``, the mean bond-length
        change (A) of the occupant pair named by ``labels``.
    error : ndarray, (n,)
        Standard error over configurations (zero for one configuration).
    count : ndarray, (n,)
        Number of contributing atom pairs per configuration.
    tensor : ndarray, (n, 3, 3) or None
        Cartesian :math:`\\langle\\mathbf{u}_i \\otimes \\mathbf{u}_j\\rangle`
        (A^2) for displacements, :math:`\\langle\\mathbf{S}_i \\otimes
        \\mathbf{S}_j\\rangle` for spins (needed for the magnetic
        recalculation); ``value`` is its trace.
    normalized : bool
        Values are correlation coefficients (``normalize=True``).

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
    normalized: bool = False

    def __len__(self):
        return len(self.value)

    def pair(self, a, b=None):
        """
        Pairs between sites named ``a`` and ``b`` (either order).

        Names match a site's full name (``"Fe1/Mn1"``) or any one of its
        occupant labels (``"Fe1"``); ``b`` defaults to ``a``.
        """

        b = a if b is None else b

        def match(name, key):
            return name == key or key in name.split("/")

        keep = np.zeros(len(self), dtype=bool)
        for i, label in enumerate(self.labels):
            s, t = label.split("-")
            keep[i] = (match(s, a) and match(t, b)) or (match(s, b) and match(t, a))
        return self.select(keep)

    def within(self, r_max, r_min=0.0):
        """Pairs with ``r_min <= distance <= r_max``."""
        return self.select((self.distance >= r_min) & (self.distance <= r_max))

    def save(self, filename):
        """
        Write to HDF5.

        Parameters
        ----------
        filename : str or Path
            Output file; overwritten if it exists.

        """

        import h5py

        with h5py.File(filename, "w") as f:
            f.attrs["quantity"] = self.quantity
            f.attrs["format"] = "rmc_discord.correlations/1"
            f.attrs["normalized"] = bool(self.normalized)
            for key in ("sites", "orbits", "dR", "r", "distance", "value", "error", "count"):
                f.create_dataset(key, data=getattr(self, key), compression="gzip")
            f.create_dataset("labels", data=np.array(self.labels, dtype=object), dtype=h5py.string_dtype())
            if self.tensor is not None:
                f.create_dataset("tensor", data=self.tensor, compression="gzip")

    @classmethod
    def load(cls, filename):
        """Read a file written by :meth:`save`."""

        import h5py

        with h5py.File(filename, "r") as f:
            data = {key: f[key][()] for key in f}
            quantity = f.attrs["quantity"]
            normalized = bool(f.attrs.get("normalized", False))
        labels = [v.decode() if isinstance(v, bytes) else str(v) for v in data.pop("labels")]
        return cls(quantity=str(quantity), labels=labels, tensor=data.pop("tensor", None),
                   normalized=normalized, **data)

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
            self.normalized,
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
            of pair vectors), sorted by pair label then distance. Self terms
            (zero distance) are excluded.

        """

        keep = self.distance > 1e-8
        if r_max is not None:
            keep &= self.distance <= r_max
        c = self.select(keep)
        pair = np.array(["-".join(sorted(label.split("-"))) for label in c.labels])
        _, pair_key = np.unique(pair, return_inverse=True)
        d_key = np.round(c.distance / tol).astype(np.int64)
        keys = np.column_stack([pair_key.ravel(), d_key])
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        w = c.count.astype(float)
        sw = np.bincount(inverse, w)
        value = np.bincount(inverse, w * c.value) / sw
        # pairs of one orbit carry the same (symmetrized or refined) value, so their
        # errors are fully correlated: the weighted mean error, not a quadrature sum
        error = np.bincount(inverse, w * c.error) / sw
        distance = np.bincount(inverse, w * c.distance) / sw
        n = np.bincount(inverse)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(inverse))[::-1]
        labels = np.array(["-".join(sorted(c.labels[i].split("-"))) for i in first])
        return {"label": labels, "distance": distance, "value": value, "error": error, "n": n}


def write_shells(shells, filename):
    """
    Write a :meth:`Correlations.shells` table as CSV.

    Columns are ``label, distance, value, error, n``.
    """

    with open(filename, "w") as f:
        f.write("label,distance,value,error,n\n")
        for row in zip(shells["label"], shells["distance"], shells["value"], shells["error"], shells["n"]):
            f.write("%s,%.6f,%.8g,%.8g,%d\n" % row)


def read_shells(filename):
    """Read a CSV written by :func:`write_shells`."""

    rows = [line.rstrip("\n").split(",") for line in open(filename)][1:]
    return {
        "label": np.array([r[0] for r in rows]),
        "distance": np.array([float(r[1]) for r in rows]),
        "value": np.array([float(r[2]) for r in rows]),
        "error": np.array([float(r[3]) for r in rows]),
        "n": np.array([int(r[4]) for r in rows]),
    }


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
    tensor = quantity in ("displacement", "spin")

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

    corr = _table(sc, quantity, channels, value, error, den, T)
    corr.normalized = bool(normalize) and quantity != "spin"
    return corr


def _table(sc, quantity, channels, value, error, den, T, names=None):
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
    if names is None:
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


def bond_lengths(supercell, states=None, symmetrize=True, r_max=None):
    r"""
    Mean bond lengths of each occupant pair, including displacements.

    :func:`correlations` reports every pair at its average-structure
    separation :math:`\mathbf{r}`. The actual bond is :math:`\mathbf{r} +
    \Delta\mathbf{u}`, :math:`\Delta\mathbf{u} = \mathbf{u}_j - \mathbf{u}_i`,
    and its mean depends on which kinds occupy the two sites (the size
    effect). With occupant indicators :math:`n_a, n_b` (primes at
    :math:`\mathbf{R} + \Delta\mathbf{R}`),

    .. math::

        \langle\Delta\mathbf{u}\rangle_{ab} =
            \frac{\sum_\mathbf{R} n_a (n_b \mathbf{u})' - (n_a \mathbf{u}) n_b'}
                 {\sum_\mathbf{R} n_a n_b'},

    and likewise :math:`\langle\Delta\mathbf{u}\,\Delta\mathbf{u}^\mathsf{T}
    \rangle_{ab}`, are cyclic cross-correlations, so every separation comes
    from FFTs as in :func:`correlations`. The mean length is taken to second
    order in the fluctuation,

    .. math::

        \langle|\mathbf{d} + \boldsymbol{\epsilon}|\rangle \approx |\mathbf{d}|
            + \frac{\operatorname{tr}\Sigma - \hat{\mathbf{d}}^\mathsf{T}
            \Sigma \hat{\mathbf{d}}}{2|\mathbf{d}|},
        \qquad \mathbf{d} = \mathbf{r} + \langle\Delta\mathbf{u}\rangle,

    with :math:`\Sigma` the covariance of :math:`\Delta\mathbf{u}`.

    Parameters
    ----------
    supercell : Supercell
        Geometry, kinds, and (if ``states`` is omitted) the configuration.
    states : iterable of dict, optional
        Configurations with keys ``kind``, ``u`` (missing keys taken from
        ``supercell``).
    symmetrize : bool, optional
        Average over the crystal's space-group operators (lengths are
        invariant; pair counts weight the average).
    r_max : float, optional
        Largest average-structure separation kept (A).

    Returns
    -------
    corr : Correlations
        Quantity ``"bond"``: ``value`` is the mean length minus the
        average-structure distance (A); ``labels`` name the occupants
        (e.g. ``"Fe1-Mn2"``); ``distance`` and ``r`` are the
        average-structure separation. Only separations inside the
        supercell's inscribed sphere (radius half its smallest width) are
        kept: beyond it the periodic image of a pair, and so its bond
        length, is not unique.

    """

    sc = supercell
    if states is None:
        states = [{}]
    channels = [(j, k) for j, ks in enumerate(sc.site_kinds) for k in ks if not sc.kinds[k].vacancy]
    if not channels:
        raise ValueError("no occupied sites for bond lengths")
    site = np.array([j for j, _ in channels])
    ckind = np.array([k for _, k in channels])
    N = np.array(sc.N)
    r0 = [_separations(sc, site[c:c + 1], site, N)[0] for c in range(len(site))]

    def inv0(P):
        return np.fft.ifftn(P, axes=(1, 2, 3)).real

    results = []
    for state in states:
        kind = np.asarray(state.get("kind", sc.kind))
        u = np.asarray(state.get("u", sc.u), dtype=float)
        gk = _grid(sc, kind[:, None])[site, ..., 0]
        on = (gk == ckind[:, None, None, None]).astype(float)
        gu = _grid(sc, u)[site] * on[..., None]
        F0 = np.fft.fftn(on, axes=(1, 2, 3))
        F1 = np.fft.fftn(gu, axes=(1, 2, 3))
        F2 = np.fft.fftn(gu[..., :, None] * gu[..., None, :], axes=(1, 2, 3))
        n_ch = len(site)
        den = np.empty((n_ch, *on.shape))
        length = np.empty_like(den)
        for c in range(n_ch):  # one row of channel pairs at a time keeps memory at O(n_ch N^3)
            a0, a1, a2 = np.conj(F0[c]), np.conj(F1[c]), np.conj(F2[c])
            dc = inv0(a0 * F0)
            m1 = inv0(a0[..., None] * F1 - a1 * F0[..., None])
            cross = inv0(a1[..., :, None] * F1[..., None, :])
            m2 = inv0(a0[..., None, None] * F2 + a2 * F0[..., None, None]) - cross - np.swapaxes(cross, -1, -2)
            ok = dc > 0.5
            safe = np.where(ok, dc, 1.0)
            mean = m1 / safe[..., None]
            cov = m2 / safe[..., None, None] - mean[..., :, None] * mean[..., None, :]
            d = r0[c] + mean
            L = np.linalg.norm(d, axis=-1)
            Ls = np.where(L > 1e-8, L, 1.0)
            dh = d / Ls[..., None]
            spread = np.trace(cov, axis1=-2, axis2=-1) - np.einsum("...a,...ab,...b->...", dh, cov, dh)
            length[c] = np.where(ok & (L > 1e-8), L + spread / (2 * Ls), 0.0)
            den[c] = np.where(ok, dc, 0.0)
        if symmetrize:
            num, den = _symmetrize_channels(sc, site, ckind, length * den, den)
            length = num / np.where(den > 0, den, 1.0)
        results.append((length, den))

    lengths = np.array([r[0] for r in results])
    length = lengths.mean(axis=0)
    error = lengths.std(axis=0, ddof=1) / np.sqrt(len(lengths)) if len(lengths) > 1 else np.zeros_like(length)
    den = np.mean([r[1] for r in results], axis=0)
    change = np.where(den > 0, length - np.linalg.norm(r0, axis=-1), 0.0)
    del r0
    corr = _table(sc, "bond", site, change, error, den, None, names=[sc.kinds[k].label for k in ckind])
    corr = corr.select((corr.distance > 1e-8) & (corr.distance < 0.5 * _width(sc) - 1e-6))
    if r_max is not None:
        corr = corr.within(r_max)
    return corr


def _width(sc):
    """Smallest perpendicular width of the supercell (A)."""
    L = sc.crystal.cartesian(np.diag(np.array(sc.N, dtype=float)))
    return 1.0 / np.max(np.linalg.norm(np.linalg.inv(L), axis=0))


def _separations(sc, site_a, site_b, N):
    """Average-structure minimum-image separations from ``site_a`` to ``site_b``, (n_a, n_b, N0, N1, N2, 3) in A."""
    G = np.stack(np.meshgrid(*[np.arange(k) for k in N], indexing="ij"), -1)
    xa = np.array([sc.sites[j].xyz for j in site_a])
    xb = np.array([sc.sites[j].xyz for j in site_b])
    dx = xb[None, :, :] - xa[:, None, :]
    frac = G[None, None] + dx[:, :, None, None, None]
    frac = frac - N * np.round(frac / N)
    return sc.crystal.cartesian(frac.reshape(-1, 3)).reshape(frac.shape)


def _symmetrize_channels(sc, site, ckind, num, den):
    """Space-group sum of scalar pair fields on (site, kind) channels, divided by the operator count."""
    unique, inverse = np.unique(site, return_inverse=True)
    maps = _site_maps(sc, unique)
    N = np.array(sc.N)
    n_ch = len(site)
    G = np.stack(np.meshgrid(*[np.arange(k) for k in N], indexing="ij"), -1).reshape(-1, 3)
    label = [sc.kinds[k].label for k in ckind]
    index = {(s, l): c for c, (s, l) in enumerate(zip(inverse, label))}
    flat_num = num.reshape(n_ch, n_ch, -1)
    flat_den = den.reshape(n_ch, n_ch, -1)
    out_num = np.zeros_like(flat_num)
    out_den = np.zeros_like(flat_den)
    for W, sigma, t, _ in maps:
        to = np.array([index[sigma[s], l] for s, l in zip(inverse, label)])
        WG = G @ W.T
        for c in range(n_ch):
            R = np.mod(WG[None] + t[inverse][:, None] - t[inverse[c]], N)
            f = np.ravel_multi_index(R.reshape(-1, 3).T, N).reshape(n_ch, -1)
            for e in range(n_ch):
                out_num[to[c], to[e], f[e]] += flat_num[c, e]
                out_den[to[c], to[e], f[e]] += flat_den[c, e]
    k = len(maps)
    return out_num.reshape(num.shape) / k, out_den.reshape(den.shape) / k
