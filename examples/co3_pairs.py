"""
Ba3Co2O6(CO3)0.7 at 50 K (CORELLI): carbonate correlations with PairModel.

The same refinement as ``co3.py``, through the workflow used by the GUI
(``from_volume`` -> ``PairModel.fit`` -> ``tables``). The partially
occupied carbonate groups (C1 + 3 O7 at z = 1/2, C2 + 3 O8 at z = +/-0.187)
are rigid units found from the occupancies; their within-unit
covariances are fixed by the average structure and set the absolute
scale. Pairs are kept to ``r_max`` in all directions and to 80 A along
the carbonate channel (the diffuse sheets are sharp along l). The data
are symmetrized to 6/mmm while the structure is P-6, so one bin per
6/mmm orbit is used and the model is averaged over the extra mirror.
Powder rings are masked and a radial background absorbs the remaining
isotropic scattering.

Three nested models are compared by held-out chi^2: occupancy only,
plus the size effect (rigid units moving along c toward or away from a
vacancy), plus displacement correlations along c.

Usage: python co3_pairs.py [r_max]   (default 12; data under $DIFFUSE_DATA_ROOT)
Writes co3_pairs/<model>_<channel>.h5 and shell CSVs.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from rmc_discord.analysis import write_shells
from rmc_discord.crystal.cif import read_cif
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import volume
from rmc_discord.pairs import Structure
from rmc_discord.pairs.sampling import from_volume
from rmc_discord.pairs.workflow import ModelSpec, PairModel

root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "Ba3Co2O6_CO3_0p7_50K"
out = Path("co3_pairs")
out.mkdir(exist_ok=True)
r_max = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0

EXTRA = ("y, x, z",)  # data symmetrized to 6/mmm, structure is P-6 (6/m)
CARBONATE = ("C1", "C2", "O7", "O8")
CHAIN = ((0, 0, 1), 3.0, 80.0)  # axis, radius, half-length (A)

t0 = time.time()
data = volume.load(next(root.glob("*_data.nxs")))
norm = volume.load(next(root.glob("*_norm.nxs")))
cif = read_cif(next(root.glob("*.cif")))
crystal = Crystal(data.lattice[:3] + (90, 90, 120), cif.space_group, cif.sites, operators=cif.operators,
                  number=cif.number)
# CIF is x = 0.6; the sample is x = 0.7
crystal = crystal.with_occupancies({s.label: min(1.0, s.occupancy * 7 / 6) for s in crystal.asymmetric
                                    if s.label in CARBONATE})
structure = Structure.from_crystal(crystal)

obs = from_volume(data / norm, structure, bragg_radius=0.2, rings=True, merge="first", extra=EXTRA,
                  held_out=0.1, seed=0)
print(f"{obs.info['bins']} measured bins, {obs.info['bragg']} near Bragg, {obs.info['rings']} on "
      f"{len(obs.info['ring_centers'])} powder rings -> {len(obs)} orbits ({obs.held.sum()} held out); "
      f"{time.time() - t0:.0f} s")

variants = {
    "occupancy": dict(),
    "size": dict(size_effect=True),
    "size+disp": dict(size_effect=True, displacement=True),
}
names = {"C1": "C1", "C2": "C2", "O7": "O7", "O8": "O8"}
print(f"\n  {'model':10s} {'coeffs':>7s} {'free':>6s} {'ties':>5s} {'chi2 train':>10s} {'chi2 test':>10s} {'time':>6s}")
results = {}
for name, kw in variants.items():
    t = time.time()
    spec = ModelSpec(r_max=r_max, cylinder=CHAIN, axis=(0, 0, 1), extra=EXTRA, **kw)
    model = PairModel(structure, spec)
    fit = model.fit(obs, background="radial", spacing=0.25)
    results[name] = (model, fit)
    print(f"  {name:10s} {model.n_coefficients:7d} {model.n_free:6d} {fit.ties:5d} {fit.chi2_train:10.3f} "
          f"{fit.chi2_test:10.3f} {time.time() - t:5.0f}s", flush=True)

best = min(results, key=lambda k: results[k][1].chi2_test)
model, fit = results[best]
print(f"\nbest by held-out chi2: {best}; scale {fit.refinement.scale:.4g} +/- {fit.refinement.scale_esd:.2g}")
tables = model.tables(fit, names)
for channel, table in tables.items():
    table.save(out / f"{best}_{channel.replace(' ', '_')}_{r_max:g}.h5")
    write_shells(table.shells(), out / f"{best}_{channel.replace(' ', '_')}_shells_{r_max:g}.csv")

# carbon-carbon correlations along the channel (Warren-Cowley alpha = C / sqrt(x_i(1-x_i) x_j(1-x_j)))
occ = {s.label: s.occupancy for s in crystal.asymmetric}
shells = tables["occupancy"].shells()
print(f"\ncarbon-carbon occupancy correlations ({best}):")
print(f"  {'pair':8s} {'r (A)':>7s} {'alpha':>8s} {'esd':>7s}")
order = np.argsort(shells["distance"])
for k in order:
    a, b = shells["label"][k].split("-")
    if a[0] != "C" or b[0] != "C" or shells["distance"][k] > 30:
        continue
    n = np.sqrt(occ[a] * (1 - occ[a]) * occ[b] * (1 - occ[b]))
    print(f"  {shells['label'][k]:8s} {shells['distance'][k]:7.3f} {shells['value'][k] / n:+8.3f} "
          f"{shells['error'][k] / n:7.3f}")
if "size effect" in tables:
    shells = tables["size effect"].shells()
    print(f"\nsize effect between carbon atoms (projection on the bond, A; > 0: the neighbor moves toward the vacancy):")
    for k in np.argsort(shells["distance"]):
        a, b = shells["label"][k].split("-")
        if a[0] == "C" and b[0] == "C" and shells["distance"][k] <= 30:
            print(f"  {shells['label'][k]:8s} {shells['distance'][k]:7.3f} {shells['value'][k]:+9.5f} "
                  f"{shells['error'][k]:8.5f}")
