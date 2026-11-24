"""
Synthetic known-answer data for Ho2Ti2O7: spin ice (magnetic) and
occupational ice (nuclear), as 3-D volumes the GUI can load.

The truth is a nearest-neighbor ice model on the Ho pyrochlore sublattice
(16 x 16 x 16 conventional cells, loop moves at T = 0.5 J) from
discord-tools, averaged over 10 chains x 8 samples so the data are nearly
speckle-free:

* ``spin_ice``: Ising moments of 10 muB along the local <111> axes
  (two-in-two-out), Ho3+ <j0> form factor; pinch points at (002), (111).
* ``occ_ice``: the same constraint read as occupancy, two of four Ho sites
  per tetrahedron occupied (b_Ho = 8.01 fm, vacancies otherwise).

Both volumes are on (h, k, l) with steps of 1/8 rlu over [-5, 5] (bin
centers on multiples of 1/8, so an 8 x 8 x 8 supercell is commensurate),
diffuse only (Bragg points NaN), with 3% multiplicative noise and matching
variance. CIFs: full Ho2Ti2O7 (Fd-3m origin 2), with Ho occupancy 1 for
spin ice and 1/2 for occupational ice.

Writes into ``$DIFFUSE_DATA_ROOT/pyrochlore_synthetic`` (default
``~/git/discordant-parsimony/local_data``): ``spin_ice.nxs``,
``occ_ice.nxs``, ``Ho2Ti2O7.cif``, ``Ho2Ti2O7_half.cif``, and
``truth.npz`` (configurations, local axes, correlations).
"""

import os
import time
from pathlib import Path

import numpy as np

from discord.material import Crystal as DTCrystal
from discord.disorder import DisorderModel, ReplicaExchange, SiteGraph, scattering
from rmc_discord.crystal.tables import magnetic_form_factor
from rmc_discord.data.volume import Volume, save

a = 10.1
N = 16
n_chains, n_samples = 10, 8
step, h_max = 1 / 8, 5.0
noise = 0.03
root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
out = root / "pyrochlore_synthetic"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

t0 = time.time()
crystal = DTCrystal([a] * 3 + [90] * 3, "F d -3 m :2", [["Ho", 0.5, 0.5, 0.5]], (N,) * 3)
graph = SiteGraph(crystal, d_cut=0.36 * a)
tets = graph.cliques([0], 4)
color, axis = graph.ice_gauge(tets)
model = DisorderModel(graph, 2)
model.set_loop_clusters(tets)
model.add_cluster_term(tets, lambda s: 1.0 * (sum(s) - 2) ** 2)
start = (axis[:, 2] > 0).astype(np.int64)

h = np.arange(-h_max, h_max + step / 2, step)
n = len(h)
hkl = np.stack(np.meshgrid(h, h, h, indexing="ij"), -1).reshape(-1, 3)
Q = 2 * np.pi * np.linalg.norm(hkl, axis=1) / a
form = magnetic_form_factor("Ho3+", Q)  # <j0> (g = 2), as the GUI uses


def measure(sig, u):
    m = (2 * sig - 1)[:, None] * axis * 10.0
    return {"Im": scattering.magnetic(graph, hkl, m, form=form),
            "In": scattering.nuclear(graph, hkl, sig, b=[0.0, 8.01])}


re = ReplicaExchange(model, [0.5] * n_chains, seed=1)
re.initialize(np.array([start] * n_chains))
res = re.run(n_samples * 5, n_thermal=50, sample_interval=5, measure=measure, exchange=False, loop=2, flip=0.2)
I = {k: v.mean(axis=(0, 1)) for k, v in res["samples"].items()}
n_in = np.array([s[tets].sum(1) for s in re.sigma])
print(f"truth: {n_chains} x {n_samples} samples, 2:2 tetrahedra {np.mean(n_in == 2):.3f} ({time.time() - t0:.0f} s)")

bragg = np.all(np.abs(hkl - np.round(hkl)) < 1e-9, axis=1)
edges = [np.linspace(-h_max - step / 2, h_max + step / 2, n + 1)] * 3
for name, key in [("spin_ice", "Im"), ("occ_ice", "In")]:
    y = I[key] * (1 + noise * rng.standard_normal(len(hkl)))
    var = (noise * I[key] + 1e-3 * I[key].mean()) ** 2
    y[bragg], var[bragg] = np.nan, np.nan
    save(Volume(signal=y.reshape(n, n, n), variance=var.reshape(n, n, n), edges=edges, W=np.eye(3)),
         out / f"{name}.nxs")
    print(f"{name}: {n}^3 points, mean {np.nanmean(y):.3g} ({time.time() - t0:.0f} s)")

cif = """data_Ho2Ti2O7
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F d -3 m :2'
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Ho1 Ho3+ 0.5 0.5 0.5 {occ} 0.003
Ti1 Ti4+ 0 0 0 1 0.003
O1 O2- 0.3302 0.125 0.125 1 0.005
O2 O2- 0.375 0.375 0.375 1 0.005
"""
(out / "Ho2Ti2O7.cif").write_text(cif.format(a=a, occ=1))
(out / "Ho2Ti2O7_half.cif").write_text(cif.format(a=a, occ=0.5))
np.savez_compressed(out / "truth.npz", sigma=re.sigma, axis=axis, r=graph.r, tets=tets,
                    I_magnetic=I["Im"], I_occupational=I["In"], hkl=hkl)
print(f"wrote {out} ({time.time() - t0:.0f} s)")
