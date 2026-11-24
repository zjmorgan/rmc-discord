r"""
Pair-correlation model of a structure: channels, constraints, fit, output.

A :class:`PairModel` turns a :class:`ModelSpec` into column blocks, one
per correlation channel:

* ``occupancy``: covariances :math:`\langle\delta n_i \delta n_j\rangle`
  per pair orbit; within-cluster values are fixed by the average
  structure (they set the absolute scale), sum rules of exclusive
  clusters, rigid units and homometric ties are imposed;
* ``size effect``: occupation-displacement vectors (first order);
* ``displacement``: symmetric displacement tensors (second order);
* ``spin``: isotropic :math:`\langle\hat{\mathbf{S}}_i\cdot
  \hat{\mathbf{S}}_j\rangle` per orbit or symmetric spin tensors, with
  unit self correlation fixing the scale (or left out for difference
  data, where the paramagnetic self term cancels).

Design columns are generated tile by tile and reduced to normal
equations (:mod:`rmc_discord.pairs.normal`), so large volumes fit in
memory; held-out observations give the validation :math:`\chi^2`.
"""

from dataclasses import dataclass, field

import numpy as np

from rmc_discord.pairs.constraints import (
    Parameterization,
    affine_parameterization,
    collapse_parameterization,
    homometric_groups,
    parameterize,
    self_trace_rows,
    stack_parameterizations,
    sum_rule_rows,
    tie_rows,
    unit_rows,
    vector_groups,
)
from rmc_discord.pairs.correlations import fixed_cluster_orbits
from rmc_discord.pairs.design import (
    design_matrix,
    displacement_design_matrix,
    magnetic_amplitude,
    magnetic_design_matrix,
    size_effect_design_matrix,
)
from rmc_discord.pairs.normal import accumulate, held_out_chi2, refine_normal
from rmc_discord.pairs.output import to_correlations
from rmc_discord.pairs.refine import background_basis, radial_background_basis
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.selection import coset_operators
from rmc_discord.pairs.symmetry import (
    coefficient_slots,
    pair_orbits,
    rigid_unit_tensor_rows,
    rigid_unit_vector_rows,
    tensor_orbits,
    vector_orbits,
)

CHANNELS = ("occupancy", "size effect", "displacement", "spin")


def _add_row(par, row, tol=1e-10):
    r"""
    Parameterization with one more homogeneous constraint ``row @ c = 0``.

    With :math:`\mathbf{c} = \mathbf{c}_0 + N\boldsymbol{\theta}` the
    constraint restricts :math:`\boldsymbol{\theta}` to the null space of
    :math:`\mathbf{r}N` (shifted to satisfy it), giving the same subspace
    as re-solving all rows. ``None`` if the row removes no free direction.
    """

    from scipy.linalg import null_space

    rN = row @ par.N
    norm2 = rN @ rN
    if norm2 <= (tol * np.linalg.norm(row)) ** 2:
        return None
    shift = -(row @ par.c0) / norm2 * rN
    return Parameterization(par.c0 + par.N @ shift, par.N @ null_space(rN[None]), par.residual)


@dataclass
class ModelSpec:
    """
    What is refined.

    Attributes
    ----------
    active : list of str
        Labels of the sites with occupational/displacive disorder
        (default: the partially occupied ones).
    r_max : float
        Pair range in Angstrom.
    cylinder : tuple or None
        ``(axis, radius, half_length)`` for long pairs along a chain.
    cluster_distance : float
        Distance (A) grouping split sites and rigid units.
    occupancy, size_effect, displacement : bool
        Channels on the active sites.
    axis : tuple or None
        Restrict displacements to a fractional direction [uvw].
    plane : tuple or None
        Restrict displacements to the lattice plane (hkl).
    magnetic : {None, "isotropic", "tensor"}
        Spin channel.
    magnetic_sites : list of str
        One label per magnetic position (e.g. ``"Fe1"`` of Fe1/Mn1).
    ions : dict
        ``{label: ion}`` or ``{label: [(fraction, ion), ...]}``.
    moments, g : dict
        ``{ion: moment}`` (Bohr magnetons) and ``{ion: g}``.
    difference : bool
        Difference data: the magnetic self term cancels and is left out.
    thermal : bool
        Debye-Waller factors from the CIF ADPs.
    extra : tuple
        Extra Laue operators of the data (e.g. ``("y, x, z",)``).
    collapse : bool
        Refine one value per set of pair vectors: orbits with the same
        vectors (e.g. C2-C2 and O8-O8 between two carbonate units, or
        Fe1-Fe1 and Fe2-Fe2 at the same separation) differ only by their
        amplitude prefactor, so only their weighted sum is determined.
        Tables report the weighted mean of each group.

    """

    active: list = None
    r_max: float = 10.0
    cylinder: tuple = None
    cluster_distance: float = 1.5
    occupancy: bool = True
    size_effect: bool = False
    displacement: bool = False
    axis: tuple = None
    plane: tuple = None
    magnetic: str = None
    magnetic_sites: list = field(default_factory=list)
    ions: dict = field(default_factory=dict)
    moments: dict = field(default_factory=dict)
    g: dict = field(default_factory=dict)
    difference: bool = False
    thermal: bool = True
    extra: tuple = ()
    collapse: bool = True


@dataclass
class Block:
    """Coefficients of one channel: columns, constraints and the pairs they live on."""

    name: str
    n: int
    parameterization: Parameterization
    design: object  # design(h, widths, projection, operators) -> (m, n)
    orbits: object
    sites: object
    kind: str  # "orbit", "vector", "tensor"
    fields: object = None
    columns: np.ndarray = None  # orbit index of each column for kind "orbit"
    fixed: tuple = None  # (index, value) of fixed columns, kind "orbit"
    rows: np.ndarray = None  # homogeneous constraint rows, kind "orbit"
    amplitude: object = None  # magnetic amplitude(Q) -> (a, a2); None for nuclear channels
    groups: np.ndarray = None  # vector group of each column (collapsed orbit channels)
    weights: np.ndarray = None  # amplitude weight of each column in its group sum

    def finish(self, par):
        """The parameterization with unresolvable vector groups collapsed."""
        if self.groups is None:
            return par
        return collapse_parameterization(par, self.groups, self.weights)


@dataclass
class FitResult:
    """A refinement of a :class:`PairModel`."""

    refinement: object
    chi2_train: float
    chi2_test: float
    n_train: int
    n_test: int
    blocks: list
    offsets: list
    background: object
    ties: int = 0

    def coefficients(self, name):
        """``(c, esd)`` of one channel."""
        for block, start in zip(self.blocks, self.offsets):
            if block.name == name:
                sl = slice(start, start + block.n)
                return self.refinement.c[sl], self.refinement.esd[sl]
        raise KeyError(name)


def group_means(groups, weights, c, esd, cov=None):
    """
    Weighted mean coefficient of each vector group, on every member.

    ``sum_k w_k c_k / sum_k w_k`` is what the data determine for a group;
    its uncertainty comes from the covariance of the members when given
    (otherwise the weighted mean of their uncertainties, as for fully
    correlated members).
    """

    labels, inverse = np.unique(groups, return_inverse=True)
    inverse = inverse.ravel()
    w = np.asarray(weights, float)
    sw = np.bincount(inverse, w)
    sw = np.where(np.abs(sw) > 0, sw, 1.0)
    M = np.zeros((len(labels), len(c)))
    M[inverse, np.arange(len(c))] = w / sw[inverse]
    mean = M @ c
    if cov is not None:
        err = np.sqrt(np.maximum(np.einsum("gk,kl,gl->g", M, cov, M), 0))
    else:
        err = np.abs(M) @ esd
    return mean[inverse], err[inverse]


class PairModel:
    """
    Pair-correlation channels of an average structure.

    Parameters
    ----------
    structure : rmc_discord.pairs.Structure
    spec : ModelSpec

    """

    def __init__(self, structure, spec):
        self.structure = structure
        self.spec = spec
        self.sites = structure.sites()
        labels = spec.active if spec.active is not None else self.sites.partial()
        self.blocks = []
        self.info = {}
        if labels and (spec.occupancy or spec.size_effect or spec.displacement):
            self._structural(list(labels))
        if spec.magnetic:
            self._magnetic()
        if not self.blocks:
            raise ValueError("no channel to refine (no active or magnetic sites)")
        self.operators = coset_operators(structure, spec.extra) if spec.extra else None

    # ------------------------------------------------------------------

    def _structural(self, labels):
        s, st = self.spec, self.structure
        active = self.sites.select(labels)
        clusters = find_clusters(st, active, s.cluster_distance)
        orbits = pair_orbits(st, active, s.r_max, cylinder=s.cylinder)
        fixed = fixed_cluster_orbits(orbits, active, clusters)
        rows = [unit_rows(orbits, clusters), sum_rule_rows(orbits, clusters),
                tie_rows(homometric_groups(orbits, active), fixed[0])]
        rows = np.vstack([r for r in rows if len(r)]) if any(len(r) for r in rows) else None
        self.clusters, self.active = clusters, active
        self.info.update(active=labels, clusters=len(clusters), orbits=orbits.n_orbits, fixed=len(fixed[0]))
        th = s.thermal
        if s.occupancy:
            par = parameterize(orbits.n_orbits, fixed, rows)
            design = lambda h, w, p, o: design_matrix(h, orbits, active, st, widths=w, projection=p, operators=o,
                                                      thermal=th)
            block = Block("occupancy", orbits.n_orbits, par, design, orbits, active, "orbit",
                          columns=np.arange(orbits.n_orbits), fixed=fixed, rows=rows)
            if s.collapse:
                reps = orbits.representatives()
                block.groups, mult = vector_groups(orbits, active, exclude=fixed[0])
                block.weights = mult * active.b[orbits.i[reps]] * active.b[orbits.j[reps]]
                block.parameterization = block.finish(par)
            self.blocks.append(block)
        if s.size_effect:
            vectors = vector_orbits(orbits, st, active, exclude=fixed[0], axis=s.axis, plane=s.plane)
            par = parameterize(len(vectors), None, rigid_unit_vector_rows(orbits, vectors, clusters))
            design = lambda h, w, p, o: size_effect_design_matrix(h, orbits, vectors, active, st, widths=w,
                                                                  projection=p, operators=o, thermal=th)
            self.blocks.append(Block("size effect", len(vectors), par, design, orbits, active, "vector", vectors))
        if s.displacement:
            tensors = tensor_orbits(orbits, st, active, exclude=fixed[0], axis=s.axis, plane=s.plane)
            par = parameterize(len(tensors), None, rigid_unit_tensor_rows(orbits, tensors, clusters))
            design = lambda h, w, p, o: displacement_design_matrix(h, orbits, tensors, active, st, widths=w,
                                                                   projection=p, operators=o, thermal=th)
            self.blocks.append(Block("displacement", len(tensors), par, design, orbits, active, "tensor", tensors))

    def _magnetic(self):
        s, st = self.spec, self.structure
        msites = self.sites.select(list(s.magnetic_sites))
        amplitude = magnetic_amplitude(msites, s.ions, s.moments, s.g)
        orbits = pair_orbits(st, msites, s.r_max)
        self_orbits = np.flatnonzero(orbits.distance < 1e-8)
        th = s.thermal
        self.info.update(magnetic_orbits=orbits.n_orbits)
        if s.magnetic == "isotropic":
            cols = np.flatnonzero(orbits.distance > 1e-8) if s.difference else np.arange(orbits.n_orbits)
            # orbits with the same pair vectors are nearly collinear (mixed-site amplitudes and ADPs
            # differ only slightly): tie them to one correlation
            ties = tie_rows(homometric_groups(orbits, msites, b=np.ones(len(msites)), adp=False), self_orbits)
            ties = ties[:, cols] if len(ties) else None
            fixed = None
            if not s.difference:
                index = np.flatnonzero(np.isin(cols, self_orbits))
                fixed = (index, np.ones(len(index)))
            par = parameterize(len(cols), fixed, ties)

            def design(h, w, p, o):
                A = magnetic_design_matrix(h, orbits, msites, st, amplitude, widths=w, projection=p, operators=o,
                                           thermal=th)
                return A[:, cols]

            block = Block("spin", len(cols), par, design, orbits, msites, "orbit", columns=cols,
                          fixed=fixed, rows=ties, amplitude=amplitude)
            if s.collapse:
                reps = orbits.representatives()[cols]
                a0 = amplitude(np.zeros(1))[0][0]  # moments p mu at Q = 0
                block.groups, mult = vector_groups(orbits, msites, exclude=cols[fixed[0]] if fixed else (),
                                                   columns=cols)
                block.weights = mult * a0[orbits.i[reps]] * a0[orbits.j[reps]]
                block.parameterization = block.finish(par)
            self.blocks.append(block)
        elif s.magnetic == "tensor":
            tensors = tensor_orbits(orbits, st, msites, exclude=self_orbits if s.difference else ())
            if s.difference:
                par = parameterize(len(tensors))
            else:
                par = affine_parameterization(len(tensors), *self_trace_rows(orbits, tensors))
            design = lambda h, w, p, o: magnetic_design_matrix(h, orbits, msites, st, amplitude, tensors=tensors,
                                                               widths=w, projection=p, operators=o, thermal=th)
            self.blocks.append(Block("spin", len(tensors), par, design, orbits, msites, "tensor", tensors,
                                     amplitude=amplitude))
        else:
            raise ValueError(f"unknown magnetic model {s.magnetic!r}")

    # ------------------------------------------------------------------

    @property
    def n_coefficients(self):
        return sum(b.n for b in self.blocks)

    @property
    def n_free(self):
        return sum(b.parameterization.n_free for b in self.blocks)

    def design(self, h, widths=None, projection=None):
        """Concatenated design columns of all channels at ``h``."""
        return np.hstack([b.design(h, widths, projection, self.operators) for b in self.blocks])

    def background(self, obs, kind="radial", spacing=0.5, degree=1):
        """Background basis ``B(h)`` over the observed |Q| range."""
        Q = np.linalg.norm(self.structure.Q(obs.h), axis=1)
        if kind == "radial":
            knots = np.arange(Q.min(), Q.max() + spacing, spacing)
            return lambda h: radial_background_basis(np.linalg.norm(self.structure.Q(h), axis=1), knots)
        if kind == "polynomial":
            # fixed |Q| normalization so tiles share one basis
            Qmax = Q.max()
            return lambda h: background_basis(np.append(np.linalg.norm(self.structure.Q(h), axis=1), Qmax),
                                              degree)[:-1]
        if kind in (None, "none"):
            return None
        raise ValueError(f"unknown background {kind!r}")

    def identifiability_ties(self, neq, tol=1e-6):
        """
        Parameterizations with orbits of indistinguishable columns tied.

        Two orbit columns of one channel whose weighted correlation over
        the observations exceeds ``1 - tol`` in magnitude cannot be
        separated by the data (e.g. orbits at the same distance whose pair
        vectors differ only slightly). With :math:`A_j \approx \rho A_i`
        only :math:`c_i + \rho c_j` is determined; the least-norm
        completion :math:`c_j = \rho c_i` is imposed (equal values for
        identical columns).

        Returns
        -------
        parameterizations : list of Parameterization
        n_ties : int
        """
        out, n_ties, start = [], 0, 0
        d = np.sqrt(np.maximum(np.diag(neq.G), 1e-300))
        for b in self.blocks:
            sl = slice(start, start + b.n)
            start += b.n
            if b.kind != "orbit" or tol <= 0:
                out.append(b.parameterization)
                continue
            G = neq.G[sl, sl]
            C = G / np.outer(d[sl], d[sl])
            fixed = set() if b.fixed is None else set(np.asarray(b.fixed[0]).tolist())
            i, j = np.nonzero(np.triu(np.abs(C) > 1 - tol, 1))
            order = np.argsort(-np.abs(C[i, j]), kind="stable")
            base = [] if b.rows is None or not len(b.rows) else list(b.rows)
            par = b.parameterization
            raw = parameterize(b.n, b.fixed, np.array(base)) if len(i) else None  # before any collapse
            for a, c in zip(i[order], j[order]):
                if a in fixed or c in fixed:
                    continue
                row = np.zeros(b.n)
                row[c], row[a] = 1, -G[a, c] / G[a, a]  # c_c = rho c_a with A_c ~ rho A_a
                trial_raw = _add_row(raw, row)
                if trial_raw is None:
                    continue
                trial = b.finish(trial_raw)
                # accept only a tie that removes one free direction and leaves every
                # refinable orbit refinable: a tie conflicting with existing ones (e.g.
                # c_a = c_c by symmetry but rho != 1) would force the orbits to zero
                free = np.linalg.norm(par.N, axis=1) > 1e-8
                if trial.n_free != par.n_free - 1 or np.any(np.linalg.norm(trial.N[free], axis=1) <= 1e-8):
                    continue
                raw, par = trial_raw, trial
            n_ties += b.parameterization.n_free - par.n_free
            out.append(par)
        return out, n_ties

    def fit(self, obs, background="radial", spacing=0.5, degree=1, ridge=0.0, tile=20000, progress=None,
            collinear=1e-6):
        """
        Refine all channels against observations.

        Parameters
        ----------
        obs : rmc_discord.pairs.sampling.Observations
        background : {"radial", "polynomial", None}
        spacing : float
            Knot spacing in inverse Angstrom of the radial background.
        degree : int
            Degree of the polynomial background.
        ridge : float
            Tikhonov damping of the free coefficients, relative to the mean
            diagonal of their normal matrix (e.g. ``1e-6`` for nearly
            collinear orbits); zero by default.
        tile : int
            Observations per design tile.
        progress : callable, optional
            ``progress(fraction)`` while accumulating.
        collinear : float
            Tie orbits whose columns correlate above ``1 - collinear``
            (:meth:`identifiability_ties`); zero to disable.

        Returns
        -------
        result : FitResult

        """

        B = self.background(obs, background, spacing, degree)
        extra = obs.info.get("extra", self.spec.extra)
        if extra:
            self.operators = coset_operators(self.structure, extra)

        def design(h):
            if progress is not None:
                design.done += len(h)
                progress(design.done / len(obs))
            return self.design(h, obs.widths, obs.projection)

        design.done = 0
        train, test = accumulate(design, obs.h, obs.y, obs.sigma, B, split=obs.held, tile=tile)
        parts, n_ties = self.identifiability_ties(train, collinear)
        par = stack_parameterizations(*parts)
        refinement = refine_normal(train, parameterization=par, ridge=ridge, relative=True)
        offsets = list(np.cumsum([0] + [b.n for b in self.blocks])[:-1])
        n_test = test.n_obs if test is not None else 0
        return FitResult(refinement, refinement.chi2 / max(1, train.n_obs),
                         held_out_chi2(refinement, test) if n_test else np.nan,
                         train.n_obs, n_test, self.blocks, offsets, B, n_ties)

    def predict(self, result, h, widths=None, projection=None):
        """Model intensity (with background) at ``h``."""
        A = self.design(h, widths, projection)
        B = result.background(h) if result.background is not None else None
        return result.refinement.predict(A, B)

    def tables(self, result, names=None):
        """
        Refined correlations as pair tables, one per channel.

        Orbit channels give the orbit coefficient of every pair (the
        weighted mean of its vector group when collapsed); vector
        and tensor channels give, per pair, the projection of the vector
        on the bond (size effect, A) or the tensor trace (A^2, or
        :math:`\\langle\\hat{\\mathbf{S}}_i\\cdot\\hat{\\mathbf{S}}_j\\rangle`).
        """

        out = {}
        for block, start in zip(self.blocks, result.offsets):
            c = result.refinement.c[start : start + block.n]
            e = result.refinement.esd[start : start + block.n]
            o = block.orbits
            quantity = {"occupancy": "occupancy", "spin": "spin", "size effect": "bond",
                        "displacement": "displacement"}[block.name]
            if block.kind == "orbit":
                if block.groups is not None:
                    cov = result.refinement.cov
                    cov = None if cov is None else cov[start : start + block.n, start : start + block.n]
                    c, e = group_means(block.groups, block.weights, c, e, cov)
                cc = np.zeros(o.n_orbits)
                ee = np.zeros(o.n_orbits)
                cc[block.columns], ee[block.columns] = c, e
                table = to_correlations(o, block.sites, self.structure, cc, ee, quantity, names)
                table = table.select(np.isin(o.orbit, block.columns))  # e.g. no self pairs for difference data
            else:
                shape = (3,) if block.kind == "vector" else (3, 3)
                index, value = coefficient_slots(block.fields, len(o), shape)
                used = index >= 0
                coef = np.where(used, c[np.maximum(index, 0)], 0.0)
                var = np.where(used, e[np.maximum(index, 0)] ** 2, 0.0)
                if block.kind == "vector":
                    r = o.d @ self.structure.A.T
                    rhat = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
                    proj = np.einsum("psa,pa->ps", value, rhat)
                else:
                    proj = np.trace(value, axis1=2, axis2=3)
                per_pair = np.sum(coef * proj, axis=1)
                per_err = np.sqrt(np.sum(var * proj**2, axis=1))
                table = to_correlations(o, block.sites, self.structure, np.zeros(o.n_orbits), None, quantity, names)
                table.value, table.error = per_pair, per_err
                keep = np.any(used, axis=1)
                table = table.select(keep)
            out[block.name] = table
        return out
