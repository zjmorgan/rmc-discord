"""
Bixbyite (Fe,Mn)2O3 (CORELLI): magnetic pair correlations, 6 K - 300 K.

The magnetic positions are the mixed Fe1/Mn1 (8b) and Fe2/Mn2 (24d)
sites, each carrying the occupancy-weighted moment (Fe3+ 5.92, Mn3+ 4.90
muB). Isotropic spin correlations <S_i.S_j> are refined per symmetry
orbit of pairs directly against the measured difference volume (no
supercell). The paramagnetic self term is the same at both temperatures
and cancels in the difference, so self orbits are left out and the
correlations are on the scale of the data; a radial background absorbs
the remaining smooth difference (phonons, Debye-Waller). The data are
m-3 symmetrized, so one bin per Laue orbit is used, and 10% of the orbits
are held out to choose the pair range.

Usage: python bixbyite.py [r_max ...]   (default 6 10 14; data under $DIFFUSE_DATA_ROOT)
Writes bixbyite_pairs/spin_<r_max>.h5 and spin_shells_<r_max>.csv.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from rmc_discord.analysis import write_shells
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import volume
from rmc_discord.pairs import Structure
from rmc_discord.pairs.sampling import from_volume
from rmc_discord.pairs.workflow import ModelSpec, PairModel

root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "bixbyite_CORELLI"
out = Path("bixbyite_pairs")
out.mkdir(exist_ok=True)
r_values = [float(v) for v in sys.argv[1:]] or [6.0, 10.0, 14.0]

crystal = Crystal.from_cif(str(root / "bixbyite.cif"))
structure = Structure.from_crystal(crystal)
occ = {s.label: s.occupancy for s in crystal.asymmetric}
ions = {
    "Fe1": [(occ["Fe1"], "Fe3+"), (occ["Mn1"], "Mn3+")],
    "Fe2": [(occ["Fe2"], "Fe3+"), (occ["Mn2"], "Mn3+")],
}
names = {"Fe1": "Fe1/Mn1", "Fe2": "Fe2/Mn2"}
RIDGE = 1e-6  # relative damping of nearly collinear orbits (e.g. Fe1-Fe1 and Fe2-Fe2 at 4.70 A)

t0 = time.time()
diff = volume.load(next(root.glob("bixbyite_006K_*"))) - volume.load(next(root.glob("bixbyite_300K_*")))
obs = from_volume(diff, structure, bragg_radius=0.15, q_range=(0.3, 7.0), merge="first",
                  error_floor=0.03, held_out=0.1, seed=0)
print(f"{obs.info['bins']} measured bins -> {len(obs)} orbits ({obs.held.sum()} held out); "
      f"{time.time() - t0:.0f} s")

for r_max in r_values:
    t = time.time()
    spec = ModelSpec(active=[], r_max=r_max, magnetic="isotropic", magnetic_sites=["Fe1", "Fe2"], ions=ions,
                     moments={"Fe3+": 5.92, "Mn3+": 4.90}, difference=True)
    model = PairModel(structure, spec)
    fit = model.fit(obs, background="radial", spacing=0.9, ridge=RIDGE)
    table = model.tables(fit, names)["spin"]
    print(f"\nr_max {r_max:g} A: {model.n_coefficients} orbits ({model.n_free} free), "
          f"train chi2/N {fit.chi2_train:.3f}, held-out chi2/N {fit.chi2_test:.3f} ({time.time() - t:.0f} s)")
    shells = table.shells()
    print(f"  {'pair':18s} {'r (A)':>7s} {'C (data scale)':>15s} {'esd':>9s}")
    order = np.argsort(shells["distance"])
    for k in order[:14]:
        print(f"  {shells['label'][k]:18s} {shells['distance'][k]:7.3f} {shells['value'][k]:15.4g} "
              f"{shells['error'][k]:9.2g}")
    table.save(out / f"spin_{r_max:g}.h5")
    write_shells(shells, out / f"spin_shells_{r_max:g}.csv")
