"""Reverse Monte Carlo refinement of a supercell against sampled data."""

from dataclasses import dataclass, field

import numpy as np

from rmc_discord.engine import kernels
from rmc_discord.model import scatterers

FIT_MODES = {
    "free": kernels.FIT_FREE,
    "scale": kernels.FIT_SCALE,
    "background": kernels.FIT_BACKGROUND,
    "none": kernels.FIT_NONE,
}

RECORDS = (
    "cost",
    "chi_sq",
    "temperature",
    "scale",
    "background",
    "move",
    "accepted",
    "delta",
)

MOVES = ("swap", "spin", "displace", "swap_displace", "exchange")


def parse_operator(op):
    """
    Integer matrix ``R`` of a reciprocal-space operator, ``h' = R h``.

    Parameters
    ----------
    op : str or array_like
        E.g. ``"k,h,l"`` or ``"-h,-k,l"``, or a 3x3 integer matrix.

    """

    if not isinstance(op, str):
        return np.asarray(op, dtype=int)
    names = {"h": np.array([1, 0, 0]), "k": np.array([0, 1, 0]), "l": np.array([0, 0, 1])}
    rows = []
    for part in op.replace(" ", "").split(","):
        row, sign, i = np.zeros(3, dtype=int), 1, 0
        while i < len(part):
            c = part[i]
            if c in "+-":
                sign = 1 if c == "+" else -1
            elif c in names:
                row += sign * names[c]
                sign = 1
            else:
                raise ValueError(f"cannot parse operator {op!r}")
            i += 1
        rows.append(row)
    if len(rows) != 3:
        raise ValueError(f"operator {op!r} needs three components")
    return np.array(rows)


def _equivalent_points(sample, crystal, operators):
    """Sample points mapped by each operator, stacked block-wise."""
    from rmc_discord.data.sampling import QSample

    N = np.array(sample.N)
    m, hkl = [], []
    for R in operators:
        h = sample.hkl @ R.T
        mr = h * N
        if not np.allclose(mr, np.round(mr), atol=1e-6):
            raise ValueError(f"operator {R.tolist()} is not commensurate with N={tuple(N)}")
        m.append(np.round(mr).astype(int))
        hkl.append(h)
    hkl = np.concatenate(hkl)
    n = len(operators)
    return QSample(
        m=np.concatenate(m), N=sample.N, hkl=hkl, Q=crystal.Q(hkl),
        y=np.tile(sample.y, n), variance=np.tile(sample.variance, n),
        bins=np.tile(sample.bins, n),
    )


@dataclass
class History:
    """
    Per-move record of a refinement.

    Attributes
    ----------
    cost, chi_sq, temperature, scale, background : list of ndarray
        Values after each move.
    move : list of ndarray
        Move type (``kernels.MOVE_*``).
    accepted : list of ndarray
        Whether each move was accepted.
    delta : list of ndarray
        Change of cost proposed by each move (NaN if no move was possible).

    """

    cost: list = field(default_factory=list)
    chi_sq: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    scale: list = field(default_factory=list)
    background: list = field(default_factory=list)
    move: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    delta: list = field(default_factory=list)

    def append(self, out):
        for key in RECORDS:
            getattr(self, key).append(out[key])

    def array(self, name):
        """Concatenated values of one record."""
        values = getattr(self, name)
        return np.concatenate(values) if values else np.empty(0)

    def acceptance(self, move=None, last=None):
        """
        Fraction of accepted moves.

        Parameters
        ----------
        move : int, optional
            Restrict to one move type.
        last : int, optional
            Restrict to the last ``last`` moves.

        """

        acc = self.array("accepted")
        kind = self.array("move")
        if last is not None:
            acc, kind = acc[-last:], kind[-last:]
        if move is not None:
            acc = acc[kind == move]
        return float(acc.mean()) if acc.size else np.nan


class Refinement:
    """
    RMC refinement state: supercell, sample, structure factors, and cost.

    Parameters
    ----------
    supercell : Supercell
        Initial configuration (modified in place).
    sample : QSample
        Observed intensity at commensurate points.
    nuclear : bool, optional
        Include nuclear scattering.
    magnetic : bool, optional
        Include magnetic scattering.
    displacive : iterable of int, optional
        Active sites whose displacements are refined. Other sites get the
        Debye-Waller factor of their ADP in their prefactors.
    radiation : {"neutron", "xray"}, optional
        Nuclear prefactor type.
    offset : ndarray or {"paramagnetic"}, optional
        Fixed term subtracted from the calculated intensity before scaling,
        e.g. ``"paramagnetic"`` for data with a high-temperature paramagnetic
        reference subtracted.
    fit : {"free", "scale", "background", "none"}, optional
        Which of scale and background are fitted in closed form.
    scale, background : float, optional
        Initial (or fixed) scale and background.
    penalty : float, optional
        Weight of the displacement covariance constraint, applied to each
        occupied kind on the displacive sites against its own ADP.
    step : float, optional
        Displacement step as a multiple of the kind's ADP Cholesky factor.
    cone : float, optional
        Maximum spin rotation per move (degrees).
    weights : dict, optional
        Relative frequency of ``"swap"``, ``"spin"``, ``"displace"``, and
        ``"swap_displace"`` moves (a swap on a displacive site that also
        steps the displacements of the incoming occupants). Move types not
        applicable to the supercell are dropped.
        ``"exchange"`` (default 0) swaps the displacements of two atoms on
        the same displacive site. With ``"displace"`` and
        ``"swap_displace"`` off, swaps and exchanges conserve each site's
        set of displacements, so start from
        :meth:`Supercell.randomize_displacements` and the displacement
        distribution stays at the ADP while only its arrangement is refined.
    seed : int, optional
        Seed for the move generator.
    background_basis : {None, "radial"}, optional
        ``"radial"`` replaces the constant background with a piecewise
        linear function of :math:`|\\mathbf{Q}|` (hat functions every
        ``knot_spacing``), fitted in closed form with the scale. Requires
        ``fit`` to be ``"free"`` or ``"background"`` (scale fixed).
    knot_spacing : float, optional
        Knot spacing of the radial background (1/A).
    symmetrize : list of str or array_like, optional
        Extra reciprocal-space operators, e.g. ``["k,h,l"]``, over which the
        model is averaged before comparison: use when the data were
        symmetrized with operations that are not in the model's Laue class
        (or for twin domains). Each must map commensurate points to
        commensurate points.
    threads : int, optional
        Number of Numba threads. Physical cores are usually best; Numba's
        default (all logical CPUs) can be slower with hyperthreading.

    """

    def __init__(
        self,
        supercell,
        sample,
        nuclear=True,
        magnetic=False,
        displacive=(),
        radiation="neutron",
        offset=None,
        fit="free",
        scale=1.0,
        background=0.0,
        penalty=1.0,
        step=0.5,
        cone=180.0,
        weights=None,
        seed=None,
        threads=None,
        background_basis=None,
        knot_spacing=0.5,
        symmetrize=None,
    ):
        if threads is not None:
            import numba

            numba.set_num_threads(int(threads))

        self.supercell = supercell
        self.full_sample = sample
        self._settings = dict(
            nuclear=nuclear,
            magnetic=magnetic,
            displacive=tuple(displacive),
            radiation=radiation,
            offset=offset if offset is None or isinstance(offset, str) else None,
            penalty=penalty,
            background_basis=background_basis,
            knot_spacing=knot_spacing,
            symmetrize=symmetrize,
        )
        self._array_offset = offset is not None and not isinstance(offset, str)
        self.nuclear = bool(nuclear)
        self.magnetic = bool(magnetic)
        self.displacive = np.array(sorted(set(displacive)), dtype=np.int64)

        self.fit_mode = FIT_MODES[fit]
        if background_basis not in (None, "radial"):
            raise ValueError(f"unknown background basis {background_basis!r}")
        self.has_basis = background_basis == "radial"
        if self.has_basis and self.fit_mode not in (kernels.FIT_FREE, kernels.FIT_BACKGROUND):
            raise ValueError("a background basis needs fit='free' or 'background'")
        Q = sample.Q_norm
        n_knots = max(2, int(np.ceil((Q.max() - Q.min()) / knot_spacing)) + 1)
        self.knots = np.linspace(Q.min(), Q.max(), n_knots)
        self.scale = float(scale)
        self.background = float(background)
        self.penalty = float(penalty)
        self.step = float(step)
        self.cone = float(cone)
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            kernels.seed(int(seed))

        sc = supercell
        operators = [np.eye(3, dtype=int)] + [parse_operator(op) for op in (symmetrize or [])]
        self.operators = operators
        self.n_ops = len(operators)
        points = _equivalent_points(sample, sc.crystal, operators) if self.n_ops > 1 else sample
        self._points_full = points
        if np.abs(points.m).max() >= np.iinfo(np.int16).max:
            raise ValueError("commensurate indices exceed int16")

        debye_waller = [j for j in range(sc.n_sites) if j not in set(self.displacive)]
        self._nuc_full = scatterers.nuclear_factors(sc, points, radiation, debye_waller)
        self._mag_full = scatterers.magnetic_factors(sc, points, debye_waller)

        if isinstance(offset, str) and offset == "paramagnetic":
            self._offset_full = scatterers.paramagnetic_intensity(sc, self._mag_full.dense())
        elif offset is None:
            self._offset_full = np.zeros(len(points))
        else:
            self._offset_full = np.tile(np.asarray(offset, dtype=float), self.n_ops)
        self.has_offset = bool(np.any(self._offset_full != 0))

        N = np.array(sample.N)
        self.tpN = 2 * np.pi / N
        self.Bm = np.ascontiguousarray(2 * np.pi * sc.crystal.B / N)
        self.A_inv = np.linalg.inv(sc.crystal.A)
        self.roots = tuple(np.exp(2j * np.pi * np.arange(n) / n) for n in N)
        xyz = np.array([site.xyz for site in sc.sites])
        self.xyz = np.ascontiguousarray(xyz, dtype=float)
        self._coef = np.zeros((kernels.MAX_TERMS, 2), dtype=complex)
        self._tt = np.full((kernels.MAX_TERMS, 2), -1, dtype=np.int64)
        self._mcoef = np.zeros((kernels.MAX_TERMS, 2))
        self._mtt = np.full((kernels.MAX_TERMS, 2), -1, dtype=np.int64)
        self._vec = np.zeros((kernels.MAX_TERMS, 3))
        self._theta = np.zeros(3)
        self._Phi_full = np.exp(2j * np.pi * xyz @ points.hkl.T)
        self.cell = np.ascontiguousarray(sc.cell, dtype=np.int64)
        self.site = np.ascontiguousarray(sc.site, dtype=np.int64)

        self.n_fit = {kernels.FIT_FREE: 2, kernels.FIT_SCALE: 1,
                      kernels.FIT_BACKGROUND: 1, kernels.FIT_NONE: 0}[self.fit_mode]
        if self.has_basis:
            self.n_fit += len(self.knots) - 1

        self.swap_sites = np.array(sc.mixed_sites, dtype=np.int64)
        self.spin_sites = np.array(
            sc.magnetic_sites if self.magnetic else [], dtype=np.int64
        )
        self.swap_displace_sites = np.intersect1d(self.swap_sites, self.displacive)
        # Phase-table move kernels avoid the per-point trigonometry of
        # displaced atoms; for pure spin refinements of undisplaced atoms the
        # general kernels are as fast.
        self.use_fused = bool(not self.magnetic or self.displacive.size > 0)

        self.n_kind = np.bincount(sc.kind, minlength=len(sc.kinds)).astype(np.int64)
        self.penalized = np.array(
            [
                k.site in set(self.displacive) and not k.vacancy and n > 0
                for k, n in zip(sc.kinds, self.n_kind)
            ],
            dtype=np.int64,
        )

        if weights is None:
            weights = {"swap": 1.0, "spin": 1.0, "displace": 1.0, "swap_displace": 1.0}
        weights = dict(weights)
        if self.swap_sites.size == 0:
            weights["swap"] = 0.0
        if self.spin_sites.size == 0:
            weights["spin"] = 0.0
        if self.displacive.size == 0 or not self.penalized.any():
            weights["displace"] = 0.0
            weights["exchange"] = 0.0
        if self.swap_displace_sites.size == 0:
            weights["swap_displace"] = 0.0
        p = np.array([weights.get(k, 0.0) for k in MOVES])
        if p.sum() <= 0:
            raise ValueError("no applicable move types")
        self.move_cdf = np.cumsum(p / p.sum())

        self.L = sc.cholesky_U()
        self.Linv = np.linalg.inv(self.L)

        self.history = History()
        self.temperature = 1.0
        self.drift = np.nan
        self.drifts = []

        self._set_points(np.arange(len(sample)))

    # ------------------------------------------------------------------
    # sample points

    def _set_points(self, indices):
        """Restrict the refinement to a subset of the full sample."""

        self.indices = np.asarray(indices)
        self.sample = self.full_sample.select(self.indices)
        n_full, n = len(self.full_sample), len(self.indices)
        self.point_indices = (
            self.indices[None, :] + n_full * np.arange(self.n_ops)[:, None]
        ).ravel()
        self.n_points = len(self.point_indices)
        m = self._points_full.m[self.point_indices]

        self.m = np.ascontiguousarray(m, dtype=np.int16)
        self.m_lo = np.ascontiguousarray(m.min(axis=0), dtype=np.int64)
        L = int((m.max(axis=0) - self.m_lo).max()) + 1
        self._E = np.zeros((kernels.MAX_TERMS, 3, L), dtype=complex)
        self.mm = np.ascontiguousarray(np.mod(m, self.full_sample.N), dtype=np.int16)
        self.Phi = np.ascontiguousarray(self._Phi_full[:, self.point_indices])
        y = np.ascontiguousarray(self.sample.y, dtype=float)
        self.w = np.ascontiguousarray(self.sample.weight, dtype=float)
        self.wy = self.w * y
        self.w_points = np.ascontiguousarray(np.tile(self.w, self.n_ops))
        self.wy_points = np.ascontiguousarray(np.tile(self.wy, self.n_ops))
        self.offset_points = np.ascontiguousarray(self._offset_full[self.point_indices])
        self.offset = self.offset_points.reshape(self.n_ops, n).mean(axis=0)
        self.sw = float(self.w.sum())
        self.sy = float(self.wy.sum())
        self.syy = float((self.wy * y).sum())
        self.dof = max(1, len(self.sample) - self.n_fit)

        self.xbuf = np.zeros(self.n_points)
        self.xobs = self.xbuf if self.n_ops == 1 else np.zeros(n)
        self._set_basis()

        self.nuc = self._nuc_full.select(self.point_indices)
        self.mag = self._mag_full.select(self.point_indices)

        self.F = np.zeros(self.n_points, dtype=complex)
        self.FM = np.zeros((self.n_points, 3), dtype=complex)
        self.recompute(track_drift=False)

    def _set_basis(self):
        """Hat-function background basis at the current points."""

        n = len(self.sample)
        K = len(self.knots)
        Q = np.clip(self.sample.Q_norm, self.knots[0], self.knots[-1])
        kb = np.clip(np.searchsorted(self.knots, Q, side="right") - 1, 0, K - 2)
        tb = (Q - self.knots[kb]) / (self.knots[kb + 1] - self.knots[kb])
        self.kb = np.ascontiguousarray(kb, dtype=np.int64)
        self.tb = np.ascontiguousarray(tb)
        if not self.has_basis:
            self.Ginv = np.zeros((1, 1))
            self.gy = np.zeros(1)
            self.yPy = 0.0
            self.partial = np.zeros((self._n_chunks(), 1))
            self.chunk_sums = np.zeros((len(self.partial), 3))
            return
        y = self.sample.y
        Bm = np.zeros((n, K))
        Bm[np.arange(n), kb] = 1 - tb
        Bm[np.arange(n), kb + 1] += tb
        G = Bm.T @ (self.w[:, None] * Bm)
        G += 1e-12 * np.trace(G) / K * np.eye(K)
        self.Ginv = np.linalg.inv(G)
        by = Bm.T @ (self.w * y)
        self.gy = self.Ginv @ by
        self.yPy = float(self.syy - by @ self.gy)
        self.partial = np.zeros((self._n_chunks(), K))
        self.chunk_sums = np.zeros((len(self.partial), 3))

    def _n_chunks(self):
        import numba

        return max(1, min(len(self.sample), 8 * numba.get_num_threads()))

    @property
    def _fused(self):
        return (self.use_fused, self.m_lo, self._E, self._coef, self._tt, self._mcoef, self._mtt, self._vec,
                self.xyz, self.tpN,
                self._theta, self.partial, self.chunk_sums)

    @property
    def _basis(self):
        return (self.has_basis, self.kb, self.tb, self.Ginv, self.gy, self.yPy, self.partial)

    def background_curve(self):
        """
        Fitted background at the current points.

        Returns
        -------
        b : ndarray, (n_points,)
            Constant background, or the radial background function.

        """

        if not self.has_basis:
            return np.full(len(self.sample), self.background)
        x = self.intensity() - self.offset
        bx = kernels.basis_sums(x, self.w, self.kb, self.tb, self.partial)
        c = self.gy - self.scale * (self.Ginv @ bx)
        return c[self.kb] * (1 - self.tb) + c[self.kb + 1] * self.tb

    def resample(self, fraction):
        """
        Refine against a random subset of the sample points.

        Parameters
        ----------
        fraction : float
            Fraction of the full sample to use; 1 restores all points.

        """

        n_full = len(self.full_sample)
        if fraction >= 1:
            self._set_points(np.arange(n_full))
            return
        n = max(self.n_fit + 1, int(round(fraction * n_full)))
        self._set_points(np.sort(self.rng.choice(n_full, n, replace=False)))

    # ------------------------------------------------------------------
    # packed kernel arguments

    @property
    def _points(self):
        return (self.m, self.mm, self.Bm, *self.roots, self.Phi)

    @property
    def _data(self):
        store = self.has_basis or self.n_ops > 1
        return (self.w_points, self.wy_points, self.offset_points, self.has_offset,
                self.xbuf, store)

    @property
    def _obs(self):
        return (self.n_ops, self.w, self.wy, self.xobs)

    @property
    def _factors(self):
        return (
            self.nuc.c, self.nuc.t, self.nuc.T,
            self.mag.c, self.mag.t, self.mag.T,
            self.nuclear, self.magnetic,
        )

    def _scaled_displacements(self):
        return np.ascontiguousarray((self.supercell.u @ self.A_inv.T) * self.tpN)

    # ------------------------------------------------------------------

    @property
    def P(self):
        """Dense nuclear prefactors at the current points."""
        P = self.nuc.dense()
        return P if self.nuclear else np.zeros_like(P)

    @property
    def M(self):
        """Dense magnetic prefactors at the current points."""
        M = self.mag.dense()
        return M if self.magnetic else np.zeros_like(M)

    @property
    def inv_n(self):
        return 1.0 / len(self.supercell)

    def _structure_factors(self):
        r"""
        Structure factors from scratch.

        Undisplaced atoms use per-site FFTs over the cell grid: for site
        :math:`j` and prefactor table :math:`t`,

        .. math::

            F(\mathbf{Q}) \mathrel{+}= \Phi_j(\mathbf{Q}) T_t(\mathbf{Q})
                G_{jt}[\mathbf{m} \bmod \mathbf{N}], \qquad
            G_{jt}[\mathbf{k}] = \sum_\mathbf{R} g_{jt}(\mathbf{R})
                e^{2\pi i \mathbf{k}\cdot\mathbf{R}/\mathbf{N}}

        where :math:`g_{jt}` holds :math:`c_k` of the atoms whose kind uses
        table :math:`t` (times the spin for the magnetic part). Displaced
        atoms are summed directly.

        Returns
        -------
        F : ndarray, complex, (n_points,)
            Nuclear structure factor.
        FM : ndarray, complex, (n_points, 3)
            Magnetic structure factor.

        """

        sc = self.supercell
        N = tuple(self.full_sample.N)
        n_cells = sc.n_cells
        mm = tuple(self.mm.astype(np.intp).T)

        F = np.zeros(self.n_points, dtype=complex)
        FM = np.zeros((self.n_points, 3), dtype=complex)

        displaced = np.any(self.su_scaled != 0, axis=1)

        def lattice_sum(values):
            field = values.reshape(N + values.shape[1:])
            axes = (0, 1, 2)
            return np.fft.ifftn(field, axes=axes) * n_cells

        for j in range(sc.n_sites):
            atoms = sc.atoms_on_site(j)
            keep = ~displaced[atoms]
            kinds = sc.kind[atoms]
            Phi = self.Phi[j]

            if self.nuclear:
                for t in np.unique(self.nuc.t[kinds]):
                    c = np.where((self.nuc.t[kinds] == t) & keep, self.nuc.c[kinds], 0)
                    if not np.any(c):
                        continue
                    G = lattice_sum(c[:, None])[..., 0][mm]
                    T = self.nuc.T[t] if t >= 0 else 1.0
                    F += Phi * T * G

            if self.magnetic:
                for t in np.unique(self.mag.t[kinds]):
                    c = np.where((self.mag.t[kinds] == t) & keep, self.mag.c[kinds], 0)
                    if not np.any(c):
                        continue
                    G = lattice_sum(c[:, None] * sc.S[atoms])[mm]
                    T = self.mag.T[t] if t >= 0 else 1.0
                    FM += (Phi * T)[:, None] * G

        if np.any(displaced):
            self._displaced_sum(np.flatnonzero(displaced), F, FM)

        return F, FM

    def _displaced_sum(self, sel, F, FM, max_bytes=64e6):
        r"""
        Add displaced atoms to the structure factors directly.

        Atoms are grouped by prefactor tables and summed in blocks; each
        block gets phase tables :math:`E_a[k, i] = e^{i k \theta_{ia}}`,
        :math:`\theta_a = 2\pi(R_a + x_a)/N_a + s_a`, so a point costs two
        complex products per atom and no trigonometric function.
        """

        sc = self.supercell
        kind = sc.kind[sel]
        cn = self.nuc.c[kind] if self.nuclear else np.zeros(len(sel), dtype=complex)
        cm = self.mag.c[kind] if self.magnetic else np.zeros(len(sel))
        keep = (cn != 0) | (cm != 0)
        sel, kind, cn, cm = sel[keep], kind[keep], cn[keep], cm[keep]
        if sel.size == 0:
            return
        theta = self.tpN * (self.cell[sel] + self.xyz[self.site[sel]]) + self.su_scaled[sel]
        L = self._E.shape[2]
        block = int(max(256, max_bytes // (48 * L)))
        tn, tm = self.nuc.t[kind], self.mag.t[kind]
        dummy_n = np.zeros(1, dtype=self.nuc.T.dtype)
        dummy_m = np.zeros(1, dtype=self.mag.T.dtype)
        for a, b in sorted(set(zip(tn.tolist(), tm.tolist()))):
            group = np.flatnonzero((tn == a) & (tm == b))
            T_n = np.ascontiguousarray(self.nuc.T[a]) if a >= 0 else dummy_n
            T_m = np.ascontiguousarray(self.mag.T[b]) if b >= 0 else dummy_m
            for start in range(0, len(group), block):
                g = group[start:start + block]
                E = np.empty((3, L, len(g)), dtype=complex)
                kernels.atom_tables(self.m_lo, np.ascontiguousarray(theta[g]), E)
                kernels.accumulate_atoms(
                    self.m, self.m_lo, E, len(g),
                    np.ascontiguousarray(cn[g], dtype=complex),
                    np.ascontiguousarray(cm[g], dtype=float),
                    np.ascontiguousarray(sc.S[sel[g]], dtype=float),
                    T_n, a >= 0, T_m, b >= 0, self.nuclear, self.magnetic, F, FM,
                )

    def recompute(self, track_drift=True):
        """
        Recompute structure factors and displacement sums from scratch.

        Parameters
        ----------
        track_drift : bool, optional
            Record the relative difference between the incrementally updated
            and the recomputed structure factors in :attr:`drift`.

        """

        sc = self.supercell
        self.su_scaled = self._scaled_displacements()
        F, FM = self._structure_factors()
        if track_drift:
            norm = max(np.abs(F).max(initial=0.0), np.abs(FM).max(initial=0.0), 1e-300)
            self.drift = max(
                np.abs(F - self.F).max(initial=0.0), np.abs(FM - self.FM).max(initial=0.0)
            ) / norm
        self.F[:] = F
        self.FM[:] = FM

        n_kinds = len(sc.kinds)
        self.su = np.zeros((n_kinds, 3))
        self.suu = np.zeros((n_kinds, 3, 3))
        self.g = np.zeros(n_kinds)
        for k in np.flatnonzero(self.penalized):
            uk = sc.u[sc.kind == k]
            self.su[k] = uk.sum(axis=0)
            self.suu[k] = uk.T @ uk
            self.g[k] = kernels.site_penalty(
                self.su[k], self.suu[k], self.n_kind[k], self.Linv[k]
            )

        self.chi_sq, self.scale, self.background = self._fit()

    def point_intensity(self):
        """
        Calculated intensity per atom at every point, symmetry equivalents
        included (blocks of ``len(sample)``).

        Returns
        -------
        I : ndarray, (n_points,)
            Unscaled intensity (fm^2).

        """

        I = np.empty(self.n_points)
        kernels.intensity(self._points, self.F, self.FM, self.inv_n, self.magnetic, I)
        return I

    def intensity(self):
        """
        Calculated intensity per atom at the current sample points, averaged
        over the symmetry equivalents.

        Returns
        -------
        I : ndarray, (n_obs,)
            Unscaled intensity (fm^2).

        """

        return self.point_intensity().reshape(self.n_ops, -1).mean(axis=0)

    def model(self):
        """Scaled model ``scale * (I - offset) + background``."""
        return self.scale * (self.intensity() - self.offset) + self.background_curve()

    def _fit(self):
        sx, sxx, sxy = kernels.sums(self.point_intensity(), self._data)
        sx, sxx, sxy = kernels._observed(sx, sxx, sxy, self._data, self._obs)
        return kernels._fit(
            self.sw, self.sy, self.syy, sx, sxx, sxy,
            self.fit_mode, self.scale, self.background, self._obs, self._basis,
        )

    @property
    def reduced_chi_sq(self):
        """Current chi^2 per degree of freedom."""
        return self.chi_sq / self.dof

    def penalty_value(self):
        """Current displacement constraint term of the cost."""
        if not self.penalized.any():
            return 0.0
        return self.penalty * self.g[self.penalized == 1].mean()

    def score(self, sample, fit="background", scale=None):
        """
        Reduced chi^2 of the current configuration on other sample points.

        Uses this refinement's scattering settings (and the same supercell
        state). For held-out validation the scale stays fixed at the
        refinement's value and only the background is refit.

        Parameters
        ----------
        sample : QSample
            Points to score, e.g. held-out points.
        fit : {"background", "free", "none", "scale"}, optional
            What is refit on ``sample``.
        scale : float, optional
            Scale to use; the current scale if omitted.

        Returns
        -------
        chi_sq : float
            Reduced chi^2 on ``sample``.

        """

        if self._array_offset:
            raise ValueError("score needs a string or no offset (array offsets are tied to the points)")
        other = Refinement(
            self.supercell, sample, fit=fit,
            scale=self.scale if scale is None else scale, background=self.background,
            **self._settings,
        )
        return other.reduced_chi_sq

    def calibrate_temperature(self, n_moves=None, acceptance=0.5):
        """
        Set the temperature from the typical cost increase of a move.

        Runs ``n_moves`` greedy moves (they are kept) and sets the
        temperature so that a median uphill move is accepted with the given
        probability. Changes of chi^2 per move scale inversely with the
        number of atoms, so a fixed starting temperature is not portable.

        Parameters
        ----------
        n_moves : int, optional
            Number of trial moves; one per atom if omitted.
        acceptance : float, optional
            Acceptance probability of a median uphill move.

        Returns
        -------
        temperature : float
            New temperature.

        """

        n_moves = len(self.supercell) if n_moves is None else int(n_moves)
        mode = self.fit_mode
        if mode == kernels.FIT_FREE and self.scale <= 0:
            # with the fitted scale clamped at zero, moves barely change
            # chi^2 and the calibrated temperature would be ~0: calibrate at
            # the scale matching model and data fluctuations instead
            self.scale = self.matched_scale()
            self.fit_mode = kernels.FIT_BACKGROUND
        try:
            history = self.run(n_moves, temperature=1e-300)
        finally:
            self.fit_mode = mode
        delta = history.array("delta")
        uphill = delta[np.isfinite(delta) & (delta > 0)]
        if uphill.size == 0:
            raise RuntimeError("no uphill moves to calibrate against")
        self.temperature = float(np.median(uphill) / np.log(1 / acceptance))
        return self.temperature

    def matched_scale(self):
        r"""
        Scale matching the fluctuations of model and data.

        :math:`s = \sqrt{y^T P y / x^T P x}` with the background (constant or
        basis) projected out by :math:`P`.

        Returns
        -------
        scale : float

        """

        x = self.intensity() - self.offset
        w, y = self.w, self.sample.y
        if self.has_basis:
            bx = kernels.basis_sums(x, w, self.kb, self.tb, self.partial)
            xPx = np.sum(w * x * x) - bx @ self.Ginv @ bx
            yPy = self.yPy
        else:
            sw = w.sum()
            xPx = np.sum(w * x * x) - np.sum(w * x) ** 2 / sw
            yPy = self.syy - self.sy**2 / sw
        return float(np.sqrt(max(yPy, 0.0) / xPx)) if xPx > 0 else 1.0

    def adapt(self, history, target=0.3, rate=1.0):
        """
        Scale displacement step and spin cone toward a target acceptance.

        Parameters
        ----------
        history : History
            Moves to measure acceptance over.
        target : float, optional
            Target acceptance of each move type.
        rate : float, optional
            Adaptation strength; the factor is ``exp(rate * (acc - target))``.

        """

        acc = history.acceptance(kernels.MOVE_DISPLACE)
        if not np.isfinite(acc):
            acc = history.acceptance(kernels.MOVE_SWAP_DISPLACE)
        if np.isfinite(acc):
            self.step = float(np.clip(self.step * np.exp(rate * (acc - target)), 1e-3, 3.0))
        acc = history.acceptance(kernels.MOVE_SPIN)
        if np.isfinite(acc):
            self.cone = float(np.clip(self.cone * np.exp(rate * (acc - target)), 1.0, 180.0))

    def run(
        self,
        n_moves,
        temperature=None,
        decay=0.0,
        recompute_every=None,
        adapt=None,
        subset=None,
        restore=True,
    ):
        """
        Run Metropolis moves.

        Parameters
        ----------
        n_moves : int
            Number of proposals.
        temperature : float, optional
            Starting temperature; continues from the last run if omitted.
        decay : float, optional
            Annealing constant; ``T`` is multiplied by ``exp(-decay)`` per
            move.
        recompute_every : int, optional
            Recompute the structure factors from scratch after this many
            moves to bound roundoff drift, recording it in :attr:`drifts`.
            Drift is ~1e-14 after thousands of moves, so this is rarely
            needed; call :meth:`recompute` to check it. Never if omitted.
        adapt : float, optional
            Target acceptance for adapting the displacement step and spin
            cone after each chunk of ``recompute_every`` moves. No adaptation
            if omitted.
        subset : float, optional
            Refine each chunk against a new random fraction of the sample
            points (minibatch). The structure factors are recomputed from
            scratch for each minibatch, so ``recompute_every`` sets the
            minibatch length and drift is not recorded.
        restore : bool, optional
            Restore all sample points at the end of a minibatch run (a full
            recompute). With False the last minibatch stays active until
            :meth:`resample` is called with 1.

        Returns
        -------
        history : History
            Per-move record of this run.

        """

        if temperature is not None:
            self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        chunk = int(recompute_every) if recompute_every else int(n_moves)
        run_history = History()
        self.drifts = []
        done = 0
        sc = self.supercell
        while done < n_moves:
            if subset is not None:
                self.resample(subset)
            n = min(chunk, n_moves - done)
            out = {key: np.empty(n) for key in RECORDS}
            out["move"] = np.empty(n, dtype=np.int8)
            out["accepted"] = np.empty(n, dtype=np.int8)

            self.temperature = kernels.run(
                n,
                self._points,
                self._data,
                self._factors,
                self.F,
                self.FM,
                self.inv_n,
                (self.sw, self.sy, self.syy, float(self.dof)),
                self.fit_mode,
                self.scale,
                self.background,
                (self.cell, self.site, self.su_scaled, sc.u, sc.kind, sc.S,
                 sc.n_sites, sc.n_cells),
                (self.move_cdf, self.swap_sites, self.spin_sites, self.displacive,
                 self.swap_displace_sites, np.cos(np.radians(self.cone))),
                (self.L, self.Linv, self.A_inv, self.tpN, self.step, self.penalty,
                 self.su, self.suu, self.g, self.n_kind, self.penalized),
                self.temperature,
                float(decay),
                tuple(out[key] for key in RECORDS),
                self._basis,
                self._obs,
                self._fused,
            )
            self.scale = float(out["scale"][-1])
            self.background = float(out["background"][-1])
            self.chi_sq = float(out["chi_sq"][-1])
            self.incremental_chi_sq = self.chi_sq
            if recompute_every and subset is None:
                self.recompute()
                self.drifts.append(self.drift)

            chunk_history = History()
            chunk_history.append(out)
            run_history.append(out)
            self.history.append(out)
            if adapt is not None:
                self.adapt(chunk_history, target=adapt)
            done += n

        if subset is not None and restore:
            self.resample(1.0)

        return run_history
