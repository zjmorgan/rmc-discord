"""Tb5Ru6Sn18: zero-parameter fcc 2:2 pseudospin model vs. data.

The participating atoms are the partially occupied Tb2, Sn1 and Sn2.
If the 2.44 A Sn1-Sn1 and 2.51 A Tb2-Sn2 contacts are forbidden, each
4b cube of Tb2/Sn1 lines holds Sn1 on one tetrahedron and Tb2 on the
other (an Ising pseudospin), every 8c cage must see exactly two
Sn-facing cubes (2:2 on every fcc tetrahedron), and each cage's single
Sn2 has one allowed position. Configurations from the family
sigma(n) = (-1)^{n_a} g(n_b) satisfy the rule for any +/-1 sequence g.

Requires the private data (``$DIFFUSE_DATA_ROOT/Tb5Ru6Sn18_293K``). Output
goes to ``results/`` in the working directory.
"""

import itertools
import time
from pathlib import Path

import numpy as np

from rmc_discord.pairs import (
    Structure,
    evaluate,
    pair_terms_from_covariance,
)
from rmc_discord.pairs.correlations import covariance_from_configurations
from rmc_discord.pairs.io import data_root, reduce_volume
from rmc_discord.pairs.rules import find_clusters, forbidden_contacts

L = 8  # cells per axis of the pseudospin lattice
REPEATS = 4  # random g sequences per axis assignment
MAX_OFFSET = 2  # correlation cutoff; the 0.25 r.l.u. grid resolves +/-2

root = data_root() / "Tb5Ru6Sn18_293K"
structure = Structure.from_cif(root / "Tb5Ru6Sn18.cif")
sites = structure.sites()
active = sites.select(sites.partial())
clusters = find_clusters(structure, active, 1.2)
limits = {("Sn", "Sn"): 2.6, ("Sn", "Tb"): 2.8, ("Tb", "Tb"): 3.0}
forbidden = forbidden_contacts(structure, active, clusters, limits)

# pseudospin geometry of the Tb2/Sn1 lines: nearest 4b cube center and
# the parity of the corner the position points to
cubes = np.array([(0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)])
shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
centers = (cubes[:, None] + shifts[None]).reshape(-1, 3)
line = np.isin(active.label, ["Tb2", "Sn1"])
center = np.zeros((len(active), 3))
parity = np.zeros(len(active), int)
for p in np.flatnonzero(line):
    c = centers[np.argmin(structure.distance(centers - active.x[p]))]
    center[p] = c
    parity[p] = np.prod(np.sign(active.x[p] - c))
is_tb2 = active.label == "Tb2"
is_sn2 = active.label == "Sn2"

R = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1)


def configuration(rng, alpha, beta):
    """Occupations (L, L, L, n) of one 2:2 state."""

    g = rng.choice((-1, 1), size=2 * L)
    occ = np.zeros(R.shape[:3] + (len(active),))
    for p in np.flatnonzero(line):
        n = np.rint(2 * (R + center[p])).astype(int)
        sigma = (-1) ** n[..., alpha] * g[n[..., beta] % (2 * L)]
        tb_here = parity[p] == sigma
        occ[..., p] = tb_here if is_tb2[p] else ~tb_here
    # Sn2 allowed only where no conflicting Tb2 is present
    occ[..., is_sn2] = 1
    for i, j, r in zip(forbidden.i, forbidden.j, forbidden.R):
        if is_sn2[i] and is_tb2[j]:
            partner = np.roll(occ[..., j], tuple(-r), axis=(0, 1, 2))
            occ[..., i] *= 1 - partner
    return occ


def check_rules(occ):
    for c in clusters:
        total = sum(
            np.roll(occ[..., m], tuple(-o), axis=(0, 1, 2))
            for m, o in zip(c.members, c.offsets)
        )
        assert np.all(total == 1), "cluster not singly occupied"
    for i, j, r in zip(forbidden.i, forbidden.j, forbidden.R):
        partner = np.roll(occ[..., j], tuple(-r), axis=(0, 1, 2))
        assert not np.any(occ[..., i] * partner), "forbidden contact"


def configurations(seed=0):
    """Occupations (m, L, L, L, n) of 2:2 states for all axis choices."""

    rng = np.random.default_rng(seed)
    configs = []
    for alpha, beta in itertools.permutations(range(3), 2):
        for _ in range(REPEATS):
            occ = configuration(rng, alpha, beta)
            check_rules(occ)
            configs.append(occ)
    return np.array(configs)


def main():
    configs = configurations()
    mean = configs.mean(axis=(0, 1, 2, 3))
    for label in ("Tb2", "Sn1", "Sn2"):
        print(
            f"{label} mean occupancy {mean[active.label == label].mean():.4f}"
        )

    volume = reduce_volume(
        *[
            next(root.glob(f"*m-3m_{tag}.nxs"))
            for tag in ("data", "norm", "bkg_data", "bkg_norm")
        ]
    )

    R_cov, C = covariance_from_configurations(configs, MAX_OFFSET)
    terms = pair_terms_from_covariance(active, R_cov, C)
    terms = terms.prune(1e-12 * np.abs(terms.w).max()).merge()
    terms = terms.bin_average(volume.widths)
    print(f"{len(terms)} pair terms")

    g = np.arange(-12, 12.001, 0.25)
    hk = np.stack(np.meshgrid(g, g, indexing="xy"), -1)
    plane = np.concatenate([hk, np.ones(hk.shape[:2] + (1,))], -1)
    start = time.perf_counter()
    model = evaluate(plane, terms, structure)
    print(
        f"evaluated {model.size} points in {time.perf_counter() - start:.1f} s"
    )
    data = volume.signal[volume.index(plane)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path("results")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, img, title in zip(
        ax, (data, model), ("data (h,k,1)", "2:2 pseudospin model (h,k,1)")
    ):
        a.imshow(
            img,
            origin="lower",
            extent=[-12, 12, -12, 12],
            vmin=np.nanpercentile(img, 1),
            vmax=np.nanpercentile(img, 99.3),
        )
        a.set_title(title)
        a.set_xlabel("h")
        a.set_ylabel("k")
    fig.tight_layout()
    fig.savefig(out / "tb5ru6sn18_ice_hk1.png", dpi=80)
    print(f"min model intensity {model.min():.3g}, max {model.max():.3g}")


if __name__ == "__main__":
    main()
