"""
Tb5Ru6Sn18 at 293 K (TOPAZ): split-site occupancy.

Fm-3m, a = 13.78 A. Three sites are partially occupied: Tb2 (1/2) and Sn1 (1/2)
on the 32f diagonals of each 4b cube, and Sn2 (1/6) on 48h. Only these atoms
are refined (swaps against vacancies on each site); the diffuse is
structural, so no magnetic term and no relaxation of the full sites. The
smooth Tb3+ paramagnetic scattering at 293 K goes into the radial background.

The data were binned at 0.25 rlu with bin centers on integers, so the
natural supercell is 4 x 4 x 4 (one data bin per commensurate point). With
``N = 8`` the data sit only on even indices; the model is then larger than
the grid can constrain.

Reduction: I = data / norm - bkg_data / bkg_norm (empty-instrument
background), symmetrized m-3m.

Usage: python tb5ru6sn18.py [seed [N]]   (data under $DIFFUSE_DATA_ROOT)
"""

import os
import sys
from pathlib import Path

import numpy as np

from rmc_discord.analysis import correlations, write_shells
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import sampling, volume
from rmc_discord.engine.anneal import Schedule, anneal, split_sample
from rmc_discord.engine.refinement import Refinement
from rmc_discord.model.supercell import Supercell

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "Tb5Ru6Sn18_293K"
out = Path(f"tb5ru6sn18_N{n}_seed{seed}")
out.mkdir(exist_ok=True)


def load(kind):
    return volume.load(next(root.glob(f"*_{kind}.nxs")))


data = load("data") / load("norm") - load("bkg_data") / load("bkg_norm")
crystal = Crystal.from_cif(str(root / "Tb5Ru6Sn18.cif"))

N = (n, n, n)
sample = sampling.commensurate(data, crystal, N)
# no Bragg mask: the reduced volume has no Bragg peaks or tails (intensity at
# allowed reflections equals the diffuse level), and integer points are dropped
sample = sample.select((sample.Q_norm >= 1.0) & (sample.Q_norm < 10.0))
sample.variance = sample.variance + (0.03 * sample.y) ** 2  # systematic error floor

pool, held = split_sample(sample, n_held=50_000, seed=0)
sc = Supercell(crystal, N, ["Tb2", "Sn1", "Sn2"], rng=seed)
ref = Refinement(sc, sample.select(pool), weights={"swap": 1.0}, background_basis="radial",
                 knot_spacing=0.5, seed=seed + 1)
result = anneal(ref, held, Schedule(rounds=60, sweeps=4, cool=0.92, batch=100_000, batch_sweeps=2,
                                    scale_floor=0.25),
                save=out / "best.npz")

corr = correlations(sc, "occupancy", normalize=True)
corr.save(out / "occupancy.h5")
write_shells(corr.shells(r_max=20.0), out / "occupancy_shells.csv")
