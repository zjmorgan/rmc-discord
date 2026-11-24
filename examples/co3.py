"""Ba3Co2O6(CO3)0.7 at 50 K: carbonate correlations from CORELLI data.

The partially occupied atoms are the carbonate groups: C1 with three O7
(z = 1/2) and C2 with three O8 (z = +/-0.187), stacked in one channel
along c. Each group is a co-occupied unit (present or absent as a
whole), found automatically from the occupancies. The structure (P-6,
Laue class 6/m) comes from the x = 0.6 CIF with the carbonate occupancy
scaled to x = 0.7; the lattice comes from the measured volume, which is
symmetrized to 6/mmm, so the model is averaged over the extra mirror.
Missing molecules along the chain are compensated by neighboring
present molecules moving along c toward the vacancy: a size effect,
refined as occupation-displacement coefficients of rigid units moving
along c. No background was subtracted and the sample has powder
contamination:
rings are detected from the radial profile and masked, and a coarse
radial background in |Q| absorbs the remaining isotropic scattering.

Usage: ``python co3.py [--q-min 6] [--free-direction]``

Requires the private data (``$DIFFUSE_DATA_ROOT/Ba3Co2O6_CO3_0p7_50K``).
Output goes to ``results/`` in the working directory.
"""

import argparse
from pathlib import Path

import numpy as np

from rmc_discord.pairs import Structure
from rmc_discord.pairs.constraints import (
    homometric_groups,
    parameterize,
    sum_rule_rows,
    tie_rows,
    unit_rows,
)
from rmc_discord.pairs.correlations import (
    fixed_cluster_orbits,
    realizability_bounds,
)
from rmc_discord.pairs.io import data_root, load_lattice, reduce_volume
from rmc_discord.pairs.participation import participation_scan
from rmc_discord.pairs.refine import radial_background_basis
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.design import (
    design_matrix,
    displacement_design_matrix,
    size_effect_design_matrix,
)
from rmc_discord.pairs.selection import (
    bragg_mask,
    coset_operators,
    in_fundamental_domain,
    powder_ring_mask,
)
from rmc_discord.pairs.symmetry import (
    coefficient_slots,
    pair_orbits,
    rigid_unit_tensor_rows,
    rigid_unit_vector_rows,
    tensor_orbits,
    vector_orbits,
)

ROOT = data_root() / "Ba3Co2O6_CO3_0p7_50K"
EXTRA = ["y, x, z"]  # data symmetrized to 6/mmm, structure is 6/m
X_CIF, X = 0.6, 0.7  # carbonate content of the CIF and of the sample
CARBONATE = ("C1", "C2", "O7", "O8")
CUTOFF = 12.0  # Angstrom, all pairs (includes neighboring channels)
# sheets at l = 0.8, 2.8, 4.8 are sharp: correlations are long-ranged
# along the carbonate channel; the 0.06 r.l.u. step resolves ~8c
CHAIN = ((0, 0, 1), 3.0, 80.0)  # axis, radius, half-length (Angstrom)
BRAGG_RADIUS = 0.2  # inverse Angstrom
KNOT_SPACING = 0.25  # inverse Angstrom, rings are masked separately
FRACTION = 0.3  # of the fundamental domain used at all
TRAIN = 1 / 3  # of the used positions
UNITS = {"C1/O7 (z = 1/2)": ["C1", "O7"], "C2/O8 (z = 0.187)": ["C2", "O8"]}


def carbonate_structure(cif, lattice):
    """CIF structure on the measured lattice with x = 0.7 carbonate."""

    structure = Structure.from_cif(cif).with_cell(lattice)
    scale = X / X_CIF
    occupancies = {
        site["label"]: min(1.0, site["occupancy"] * scale)
        for site in structure.asymmetric
        if site["label"] in CARBONATE
    }
    return structure.with_occupancies(occupancies)


def report(models):
    print(
        f"  {'participating':36s} {'params':>6s} {'chi2 train':>10s}"
        f" {'chi2 test':>10s} {'wR test':>8s}"
    )
    for m in models:
        name = " + ".join(m.units) if m.units else "none (independent units)"
        print(
            f"  {name:36s} {m.n_params:6d} {m.chi2_train:10.3f}"
            f" {m.chi2_test:10.3f} {m.wR_test:8.4f}"
        )


def main(q_min=0.0, axis=(0, 0, 1)):
    data = next(ROOT.glob("*_data.nxs"))
    norm = next(ROOT.glob("*_norm.nxs"))
    cif = next(ROOT.glob("*.cif"))
    structure = carbonate_structure(cif, load_lattice(data))
    print(f"lattice from data: {np.round(structure.cell, 4)}")

    sites = structure.sites()
    active = sites.select(sites.partial())
    clusters = find_clusters(structure, active, 1.5)
    for c in clusters:
        labels = "+".join(dict.fromkeys(active.label[c.members]))
        print(
            f"  unit {labels:6s} {c.kind:12s} {len(c)} atoms,"
            f" occupancy {1 - c.vacancy:.3f}"
        )

    orbits = pair_orbits(structure, active, CUTOFF, cylinder=CHAIN)
    fixed = fixed_cluster_orbits(orbits, active, clusters)
    group = homometric_groups(orbits, active)
    rows = np.vstack(
        [
            unit_rows(orbits, clusters),
            sum_rule_rows(orbits, clusters),
            tie_rows(group, fixed[0]),
        ]
    )
    print(
        f"{orbits.n_orbits} orbits within {CUTOFF} A or {CHAIN[2]} A along c,"
        f" {len(fixed[0])} fixed"
    )

    volume = reduce_volume(data, norm)
    index = np.nonzero(np.isfinite(volume.signal) & (volume.sigma > 0))
    keep = np.zeros(len(index[0]), bool)
    for chunk in np.array_split(np.arange(len(keep)), 40):
        h = volume.hkl(tuple(i[chunk] for i in index))
        keep[chunk] = in_fundamental_domain(h, structure, EXTRA)
    index = tuple(i[keep] for i in index)
    h = volume.hkl(index)
    y = volume.signal[index]
    Q = np.linalg.norm(structure.Q(h), axis=1)

    bragg = bragg_mask(h, structure, BRAGG_RADIUS)
    ring, rings = powder_ring_mask(Q[~bragg], y[~bragg])
    on_ring = np.zeros(len(h), bool)
    on_ring[~bragg] = ring
    print(
        f"powder rings: {len(rings)} shells masked"
        f" ({on_ring.mean():.1%} of positions)"
    )

    rng = np.random.default_rng(0)
    keep = ~bragg & ~on_ring & (Q >= q_min)
    keep &= rng.random(len(h)) < FRACTION
    index, h, Q = tuple(i[keep] for i in index), h[keep], Q[keep]
    y, sigma = volume.signal[index], volume.sigma[index]
    print(
        f"{len(h)} positions used, |Q| in [{Q.min():.1f}, {Q.max():.1f}] 1/A"
    )

    W = coset_operators(structure, EXTRA)
    A = design_matrix(
        h,
        orbits,
        active,
        structure,
        widths=volume.widths,
        operators=W,
        projection=volume.W,
    )
    kw = dict(widths=volume.widths, operators=W, projection=volume.W)
    vectors = vector_orbits(
        orbits, structure, active, exclude=fixed[0], axis=axis
    )
    vector_rows = rigid_unit_vector_rows(orbits, vectors, clusters)
    tensors = tensor_orbits(
        orbits, structure, active, exclude=fixed[0], axis=axis
    )
    tensor_rows = rigid_unit_tensor_rows(orbits, tensors, clusters)
    where = "along c" if axis is not None else "in any allowed direction"
    print(
        f"displacements {where}: {len(vectors)} size-effect and"
        f" {len(tensors)} displacement-correlation coefficients,"
        f" {parameterize(len(vectors), None, vector_rows).n_free} and"
        f" {parameterize(len(tensors), None, tensor_rows).n_free} free"
        " after rigid-unit ties"
    )
    A_size = size_effect_design_matrix(
        h, orbits, vectors, active, structure, **kw
    )
    A_disp = displacement_design_matrix(
        h, orbits, tensors, active, structure, **kw
    )
    mechanisms = [
        {
            "name": "size effect",
            "A": A_size,
            "orbit": vectors.orbit,
            "rows": vector_rows,
        },
        {
            "name": "displacements",
            "A": A_disp,
            "orbit": tensors.orbit,
            "rows": tensor_rows,
        },
    ]
    knots = np.arange(Q.min(), Q.max() + KNOT_SPACING, KNOT_SPACING)
    B = radial_background_basis(Q, knots)
    train = rng.random(len(h)) < TRAIN

    models = participation_scan(
        y,
        sigma,
        A,
        orbits,
        active,
        UNITS,
        fixed,
        rows,
        train,
        B,
        mechanisms=mechanisms,
    )
    print("participation scan (held-out positions):")
    report(models)

    best = models[0].refinement
    print(f"scale = {best.scale:.4g} +/- {best.scale_esd:.2g}")
    lower, upper = realizability_bounds(orbits, active, clusters)
    reps = orbits.representatives()
    print("refined carbon-carbon correlations (alpha = C / x(1 - x)):")
    for o in np.flatnonzero(best.free[: orbits.n_orbits]):
        p = reps[o]
        li, lj = active.label[orbits.i[p]], active.label[orbits.j[p]]
        if li[0] != "C" or lj[0] != "C" or orbits.distance[o] > 30:
            continue
        xi = active.occupancy[orbits.i[p]]
        xj = active.occupancy[orbits.j[p]]
        norm_ = np.sqrt(xi * (1 - xi) * xj * (1 - xj))
        flag = ""
        if best.c[o] < lower[o] - 2 * best.esd[o]:
            flag = f"  below bound {lower[o] / norm_:+.3f}"
        elif best.c[o] > upper[o] + 2 * best.esd[o]:
            flag = f"  above bound {upper[o] / norm_:+.3f}"
        print(
            f"  {li}-{lj} {orbits.distance[o]:6.3f} A:"
            f" alpha = {best.c[o] / norm_:+.3f} +/- {best.esd[o] / norm_:.3f}"
            f"{flag}"
        )
    # coefficient blocks of the best model, in mechanism order
    offset, blocks = orbits.n_orbits, {}
    for mech in mechanisms:
        if mech["name"] in models[0].units:
            n_m = mech["A"].shape[1]
            blocks[mech["name"]] = (offset, n_m)
            offset += n_m

    def carbon_pairs(fields, shape):
        index, value = coefficient_slots(fields, len(orbits), shape)
        for o in np.unique(fields.orbit):
            p = reps[o]
            li, lj = active.label[orbits.i[p]], active.label[orbits.j[p]]
            if li[0] == "C" and lj[0] == "C" and orbits.distance[o] <= 30:
                ks = [k for k in index[p] if k >= 0]
                slots = [list(index[p]).index(k) for k in ks]
                yield o, p, li, lj, ks, [value[p, q] for q in slots]

    if "size effect" in blocks:
        o0, _ = blocks["size effect"]
        print("refined size effect between carbon atoms (sigma . d_hat > 0:")
        print("  the neighbor moves toward the vacancy):")
        for o, p, li, lj, ks, vals in carbon_pairs(vectors, (3,)):
            vec = sum(best.c[o0 + k] * v for k, v in zip(ks, vals))
            err = np.sqrt(sum(best.esd[o0 + k] ** 2 for k in ks))
            dhat = orbits.d[p] @ structure.A.T
            dhat = dhat / np.linalg.norm(dhat)
            print(
                f"  {li}-{lj} {orbits.distance[o]:6.3f} A:"
                f" sigma.d_hat = {vec @ dhat:+.5f} +/- {err:.5f} A"
            )
    if "displacements" in blocks:
        o0, _ = blocks["displacements"]
        print("refined displacement correlations between carbon atoms:")
        for o, p, li, lj, ks, vals in carbon_pairs(tensors, (3, 3)):
            V = sum(best.c[o0 + k] * v for k, v in zip(ks, vals))
            err = np.sqrt(sum(best.esd[o0 + k] ** 2 for k in ks))
            print(
                f"  {li}-{lj} {orbits.distance[o]:6.3f} A:"
                f" V_zz = {V[2, 2]:+.5f} +/- {err:.5f} A^2"
            )
    free = best.free[: orbits.n_orbits]
    best_c, best_esd = best.c[: orbits.n_orbits], best.esd[: orbits.n_orbits]
    outside = (best_c < lower - 2 * best_esd) | (best_c > upper + 2 * best_esd)
    print(
        f"{np.sum(outside & free)} of {free.sum()} refined coefficients"
        " lie outside the 0/1 realizability bounds (> 2 esd)"
    )

    A_best = np.hstack(
        [A] + [m["A"] for m in mechanisms if m["name"] in models[0].units]
    )
    out = Path("results")
    out.mkdir(exist_ok=True)
    np.savez(
        out / f"ba3co2o6_co3_scan_qmin{q_min:g}.npz",
        c=best.c,
        esd=best.esd,
        distance=orbits.distance,
        h=h,
        y=y,
        sigma=sigma,
        model=best.predict(A_best, B),
        background=B @ best.background,
        train=train,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--q-min", type=float, default=0.0)
    parser.add_argument(
        "--free-direction",
        action="store_true",
        help="allow displacements in every symmetry-allowed direction",
    )
    args = parser.parse_args()
    main(args.q_min, None if args.free_direction else (0, 0, 1))
