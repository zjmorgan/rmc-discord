"""
Bixbyite (Fe,Mn)2O3 (CORELLI): magnetic correlations from 6 K - 300 K.

Spins only, paramagnetic offset, full measured volume, 10 x 10 x 10
supercell, minibatches, and held-out validation.

Usage: python bixbyite.py [seed]   (data under $DIFFUSE_DATA_ROOT)
"""

import os
import sys
from pathlib import Path

from rmc_discord.analysis import correlations, write_shells
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import sampling, volume
from rmc_discord.engine.anneal import Schedule, anneal, split_sample
from rmc_discord.engine.refinement import Refinement
from rmc_discord.model.supercell import Supercell

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "bixbyite_CORELLI"
out = Path(f"bixbyite_seed{seed}")
out.mkdir(exist_ok=True)

N = (10, 10, 10)
crystal = Crystal.from_cif(str(root / "bixbyite.cif"))
diff = volume.load(next(root.glob("bixbyite_006K_*"))) - volume.load(next(root.glob("bixbyite_300K_*")))
sample = sampling.commensurate(diff, crystal, N)
sample = sample.select(~sampling.bragg_mask(sample, crystal, radius=0.15))
sample.variance = sample.variance + (0.03 * sample.y) ** 2

pool, held = split_sample(sample, held_out=0.1, seed=0)
sc = Supercell(crystal, N, ["Fe1", "Fe2"],
               ions={"Fe1": "Fe3+", "Mn1": "Mn3+", "Fe2": "Fe3+", "Mn2": "Mn3+"},
               moments={"Fe3+": 5.92, "Mn3+": 4.90}, rng=seed)
ref = Refinement(sc, sample.select(pool), nuclear=False, magnetic=True, offset="paramagnetic",
                 cone=60, weights={"spin": 1.0, "swap": 0.0}, seed=seed + 1)
result = anneal(ref, held, Schedule(rounds=40, sweeps=2, cool=0.9, temperature=None,
                                    batch=100_000, batch_sweeps=1, adapt=0.3), save=out / "best.npz")

corr = correlations(sc, "spin")
corr.save(out / "spin.h5")
write_shells(corr.shells(r_max=20.0), out / "spin_shells.csv")
