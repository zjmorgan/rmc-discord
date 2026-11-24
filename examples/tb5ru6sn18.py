"""Tb5Ru6Sn18: which partially occupied atoms participate?

Candidate units are the split-site cluster types: the Tb2/Sn1 lines and
the Sn2 cages. Every combination is refined on a random 10% of the
independent positions and scored on the rest. The scan is run on two
synthetic truths with known answers and then on the measured data.
"""

import numpy as np

from rmc_discord.pairs.constraints import (
    homometric_groups,
    sum_rule_rows,
    tie_rows,
)
from rmc_discord.pairs.correlations import (
    covariance_from_configurations,
    fixed_cluster_orbits,
)
from rmc_discord.pairs.io import reduce_volume
from rmc_discord.pairs.participation import participation_scan
from rmc_discord.pairs.refine import background_basis
from rmc_discord.pairs.design import design_matrix
from rmc_discord.pairs.selection import bragg_mask, canonical
from rmc_discord.pairs.symmetry import orbit_average, pair_orbits

from tb5ru6sn18_ice import (
    MAX_OFFSET,
    active,
    clusters,
    configurations,
    root,
    structure,
)

CUTOFF = 14.0
H_MAX = 12
BRAGG_RADIUS = 0.14
NOISE = 0.03
UNITS = {"Tb2/Sn1 lines": ["Tb2", "Sn1"], "Sn2 cages": ["Sn2"]}


def randomize_cages(configs, rng):
    """Replace each cage's Sn2 choice by an independent uniform one."""

    configs = configs.copy()
    shape = configs.shape[:4]
    for cluster in clusters:
        if active.label[cluster.members[0]] != "Sn2":
            continue
        choice = rng.integers(0, len(cluster), size=shape)
        for k, (a, o) in enumerate(zip(cluster.members, cluster.offsets)):
            configs[..., a] = np.roll(choice == k, tuple(o), axis=(1, 2, 3))
    return configs


def report(title, models):
    print(f"\n{title}")
    print(
        f"  {'participating':32s} {'params':>6s} {'chi2 train':>10s}"
        f" {'chi2 test':>10s} {'wR test':>8s}"
    )
    for m in models:
        name = (
            " + ".join(m.units) if m.units else "none (independent clusters)"
        )
        print(
            f"  {name:32s} {m.n_params:6d} {m.chi2_train:10.3f}"
            f" {m.chi2_test:10.3f} {m.wR_test:8.4f}"
        )


def main():
    rng = np.random.default_rng(4)
    orbits = pair_orbits(structure, active, CUTOFF)
    fixed = fixed_cluster_orbits(orbits, active, clusters)
    group = homometric_groups(orbits, active)
    rows = np.vstack(
        [sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])]
    )

    volume = reduce_volume(
        *[
            next(root.glob(f"*m-3m_{tag}.nxs"))
            for tag in ("data", "norm", "bkg_data", "bkg_norm")
        ]
    )
    index = np.nonzero(np.isfinite(volume.signal) & (volume.sigma > 0))
    h = volume.hkl(index)
    keep = np.all(np.abs(h) <= H_MAX, axis=1)
    keep &= ~bragg_mask(h, structure, BRAGG_RADIUS)
    index, h = tuple(i[keep] for i in index), h[keep]
    rep, first = np.unique(canonical(h, structure), axis=0, return_index=True)
    index = tuple(i[first] for i in index)
    A = design_matrix(rep, orbits, active, structure, widths=volume.widths)
    B = background_basis(np.linalg.norm(structure.Q(rep), axis=1), degree=2)
    train = rng.random(len(rep)) < 0.10
    print(
        f"{orbits.n_orbits} orbits, {len(rep)} independent positions,"
        f" {train.sum()} for training"
    )

    configs = configurations()
    for title, occ in (
        ("synthetic truth: 2:2 states (lines and cages correlated)", configs),
        (
            "synthetic truth: 2:2 lines, independent random cages",
            randomize_cages(configs, rng),
        ),
    ):
        R, C = covariance_from_configurations(occ, MAX_OFFSET)
        c = orbit_average(orbits, R, C)
        c[fixed[0]] = fixed[1]
        y0 = A @ c + B @ np.array([0.02, 0.01, 0.0]) * np.abs(A @ c).max()
        sigma = NOISE * (np.median(np.abs(y0)) + np.abs(y0))
        y = y0 + sigma * rng.normal(size=len(y0))
        models = participation_scan(
            y, sigma, A, orbits, active, UNITS, fixed, rows, train, B
        )
        report(title, models)

    y, sigma = volume.signal[index], volume.sigma[index]
    models = participation_scan(
        y, sigma, A, orbits, active, UNITS, fixed, rows, train, B
    )
    report("measured Tb5Ru6Sn18, 293 K", models)


if __name__ == "__main__":
    main()
