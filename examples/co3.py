"""
Ba3Co2O6(CO3)0.7 at 50 K (CORELLI): carbonate occupancy and displacement.

Carbonate-only rigid groups (C1/O7, C2/O8) in an 8 x 8 x 20 supercell, swap
and displacement-exchange moves (displacements stay at the ADP), radial
background, model averaged over the extra mirror of the 6/mmm-symmetrized
data, minibatches of a training pool enriched in the diffuse sheets, and
held-out validation on and off the sheets.

Usage: python co3.py [seed]   (data under $DIFFUSE_DATA_ROOT)
"""

import os
import sys
from pathlib import Path

import numpy as np

from rmc_discord.analysis import correlations, write_shells
from rmc_discord.crystal.cif import read_cif
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import sampling, volume
from rmc_discord.engine.anneal import Schedule, anneal, split_sample
from rmc_discord.engine.refinement import Refinement
from rmc_discord.model.supercell import Supercell

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "Ba3Co2O6_CO3_0p7_50K"
out = Path(f"co3_seed{seed}")
out.mkdir(exist_ok=True)

N = (8, 8, 20)
data = volume.load(next(root.glob("*_data.nxs")))
norm = volume.load(next(root.glob("*_norm.nxs")))
cif = read_cif(next(root.glob("*.cif")))
crystal = Crystal(data.lattice[:3] + (90, 90, 120), cif.space_group, cif.sites,
                  operators=cif.operators, number=cif.number)
crystal = crystal.with_occupancies({s.label: min(1.0, s.occupancy * 7 / 6) for s in crystal.asymmetric
                                    if s.label in ("C1", "C2", "O7", "O8")})

sample = sampling.commensurate(data, crystal, N, norm=norm)
bragg = sampling.bragg_mask(sample, crystal, radius=[0.15, 0.15, 0.3], allowed_only=False)
ring, _ = sampling.powder_ring_mask(sample, exclude=bragg)
sample = sample.select(~bragg & ~ring)
sample = sample.select((sample.Q_norm >= 2.5) & (sample.Q_norm < 10))
sample.variance = sample.variance + (0.03 * sample.y) ** 2  # systematic error floor

l = np.abs(sample.hkl[:, 2])
sheet = np.min(np.abs(l[:, None] - np.array([0.8, 2.85, 4.9])), axis=1) <= 0.1
pool, held = split_sample(sample, strata={"on": sheet, "off": ~sheet}, n_held=50_000, seed=0)
rng = np.random.default_rng(0)
on, off = pool[sheet[pool]], rng.permutation(pool[~sheet[pool]])
pool = np.sort(np.concatenate([on, off[: 3 * len(on)]]))  # sheets at 25% of the pool

sc = Supercell(crystal, N, ["C1", "C2"], groups={"C1": ["O7"], "C2": ["O8"]}, rng=seed)
sc.randomize_displacements()
ref = Refinement(sc, sample.select(pool), displacive=range(sc.n_sites), penalty=10.0,
                 weights={"swap": 1.0, "exchange": 1.0}, background_basis="radial",
                 knot_spacing=0.5, symmetrize=["k,h,l"], seed=seed + 1)
result = anneal(ref, held, Schedule(rounds=60, sweeps=4, cool=0.92, temperature=0.01,
                                    batch=100_000, batch_sweeps=2, scale_floor=0.25),
                save=out / "best.npz")

for quantity in ("occupancy", "displacement"):
    corr = correlations(sc, quantity, normalize=True)
    corr.save(out / f"{quantity}.h5")
    write_shells(corr.shells(r_max=20.0), out / f"{quantity}_shells.csv")
