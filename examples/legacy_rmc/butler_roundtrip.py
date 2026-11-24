"""
Round trip on known-answer data: the Butler, Withers & Welberry (1992)
size-effect alloy.

Truth configurations come from discord-tools (``examples/butler_size_effect.py``
there): a random 50/50 alloy on a square lattice, a0 = 1 A, with nearest-
neighbor rest lengths a0 (1 + 0.05), a0, a0 (1 - 0.05) for AA, AB, BB, relaxed
by 50 zero-temperature Monte Carlo steps. A = Zn (large), B = Cu.

Their exact (hk0) diffuse intensity, averaged over ``n_truth`` independent
configurations and computed with this package's own forward model, is the
data (2% noise). A refinement then starts from a random occupancy and Gaussian
displacements with the truth ADPs and is scored on held-out points. Checks:

* occupancy short-range order stays near zero (the truth is random);
* the species-resolved displacement correlations <x_0 x_lm> of Table 1 (and
  <x_0 y_lm>) are recovered: sign, magnitude, and range.

Usage: PYTHONPATH=src python examples/butler_roundtrip.py [seed [rounds [reheats]]]
Needs discord-tools (``discord``) importable.
"""

import os
import sys
import time

import numpy as np

from discord.material import Crystal as TruthCrystal
from discord.disorder import DisorderModel, ReplicaExchange, SiteGraph
from discord.disorder import correlations as truth_correlations

from rmc_discord.crystal.cif import AsymmetricSite
from rmc_discord.crystal.structure import Crystal
from rmc_discord.data.sampling import QSample
from rmc_discord.engine.anneal import Schedule, anneal, split_sample
from rmc_discord.engine.refinement import Refinement
from rmc_discord.model.supercell import Supercell

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
L = 64
n_truth = int(os.environ.get("N_TRUTH", 8))
extent = 4.5  # |h|, |k| <= extent
noise = 0.02
t0 = time.time()

# --- truth (discord-tools) -------------------------------------------------
crystal_t = TruthCrystal([1, 1, 10, 90, 90, 90], "P 4/m m m", [["Cu", 0, 0, 0]], (L, L, 1))
graph = SiteGraph(crystal_t, d_cut=1.1)
model = DisorderModel(graph, 2)
model.set_springs([0], k_long=1.0, epsilon=[[-0.05, 0], [0, 0.05]])
model.set_free_axes([1, 1, 0])
sigma0 = np.array([model.random_occupancy([0.5, 0.5], 100 + r) for r in range(n_truth)])
re = ReplicaExchange(model, [1e-9] * n_truth, seed=7)
re.initialize(sigma0)
re.step[:] = 0.02
re.run(50, displace=1, exchange=False)
truth_sigma, truth_u = np.array(re.sigma), np.array(re.u)  # species 1 = A = Zn
flat_s, flat_u = truth_sigma.ravel(), truth_u.reshape(-1, 3)
ms = np.array([[np.mean(flat_u[flat_s == k, a] ** 2) for k in (0, 1)] for a in range(3)])
print(f"truth: {n_truth} x {L}^2, <x^2> Zn {ms[0, 1] * 1e4:.1f} Cu {ms[0, 0] * 1e4:.1f} x 1e-4 A^2 "
      f"({time.time() - t0:.0f} s)")

# --- the same structure in rmc-discord ----------------------------------------
def U(k):
    return np.diag([ms[0, k], ms[1, k], 1e-6])


crystal = Crystal(
    (1, 1, 10, 90, 90, 90), "P 4/m m m",
    [AsymmetricSite("Zn1", "Zn", np.zeros(3), 0.5, U=U(1), U_iso=ms[0, 1]),
     AsymmetricSite("Cu1", "Cu", np.zeros(3), 0.5, U=U(0), U_iso=ms[0, 0])],
)
N = (L, L, 1)


def supercell(rng):
    return Supercell(crystal, N, ["Zn1"], rng=rng)


sc = supercell(seed)
label_kind = {k.label: i for i, k in enumerate(sc.kinds)}
# atom order: discord-tools index of each rmc-discord atom, by cell
cells_t = np.round(graph.r[:, :2]).astype(int) % L
order_t = np.empty(L * L, int)
order_t[cells_t[:, 0] * L + cells_t[:, 1]] = np.arange(L * L)
perm = order_t[sc.cell[:, 0] * L + sc.cell[:, 1]]


def load(sc, s, u):
    sc.kind[:] = np.where(s[perm] == 1, label_kind["Zn1"], label_kind["Cu1"])
    sc.u[:] = u[perm]
    return sc


def to_truth(sc):
    s = np.empty(L * L, int)
    u = np.empty((L * L, 3))
    s[perm] = (sc.kind == label_kind["Zn1"]).astype(int)
    u[perm] = sc.u
    return s, u


# all commensurate (hk0) points with |h|, |k| <= extent, integer points excluded
r = np.arange(-int(extent * L), int(extent * L) + 1)
m = np.stack(np.meshgrid(r, r, [0], indexing="ij"), -1).reshape(-1, 3)
m = m[~np.all(m % np.array(N) == 0, axis=1)]
hkl = m / np.array(N)
sample = QSample(m=m, N=N, hkl=hkl, Q=crystal.Q(hkl), y=np.zeros(len(m)),
                 variance=np.ones(len(m)), bins=np.ones(len(m), int))

t = supercell(0)
I = np.mean([Refinement(load(t, s, u), sample, displacive=[0], fit="none").intensity()
             for s, u in zip(truth_sigma, truth_u)], axis=0)
rng = np.random.default_rng(1)
sample.y = I * (1 + noise * rng.standard_normal(len(I)))
sample.variance = (noise * I + 0.01 * I.mean()) ** 2
print(f"data: {len(sample)} points, mean I {I.mean():.3g} fm^2 ({time.time() - t0:.0f} s)")

# --- refinement ------------------------------------------------------------
pool, held = split_sample(sample, held_out=0.1, seed=seed)
warm = os.environ.get("START") == "truth"  # diagnostic: start from a truth configuration
fix = os.environ.get("FIX")  # diagnostic: "occupancy" or "displacements" held at the truth
if warm:
    load(sc, truth_sigma[0], truth_u[0])
elif fix == "occupancy":
    load(sc, truth_sigma[0], truth_u[0])
    sc.randomize_displacements()
elif fix == "displacements":
    load(sc, truth_sigma[0], truth_u[0])
    sc.randomize_occupancy()
else:
    sc.randomize_displacements()
ref = Refinement(sc, sample.select(pool), displacive=[0], fit="none", penalty=100.0, step=0.3,
                 weights={"displace": 1.0} if fix == "occupancy" else {"swap": 1.0} if fix == "displacements"
                 else {"swap": 1.0, "displace": 1.0, "swap_displace": 1.0}, seed=seed + 1)
# reference: one truth configuration (it carries its own speckle, so even it
# cannot reach chi2 ~ 1 against the average of n_truth configurations)
reference = {}
for name, h in held.items():
    tr = Refinement(load(supercell(0), truth_sigma[0], truth_u[0]), h, displacive=[0], fit="background")
    reference[name] = tr.reduced_chi_sq
print("one truth configuration, held-out chi2: " + ", ".join(f"{k} {v:.4g}" for k, v in reference.items()))
n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
n_reheat = int(sys.argv[3]) if len(sys.argv) > 3 else 3
T_reheat = 0.05  # swaps accept ~25% here: the occupancy can rearrange again


def quick(label):
    s_, u_ = to_truth(sc)
    d = 2 * s_.reshape(L, L) - 1.0
    a10 = np.mean(d * np.roll(d, 1, axis=0))
    j = truth_correlations.displacement_pair_correlation(graph, s_, u_, (0, 0))[1]
    print(f"{label}: <s_0 s_10> {a10:+.3f} (truth ~0), [] (1,0) {j[1, 0, 0] * 1e4:.2f} "
          f"(4,0) {j[4, 0, 0] * 1e4:.2f} x 1e-4 ({time.time() - t0:.0f} s)")


if os.environ.get("DISP_FIRST"):
    # displacements first against the fixed random occupancy, then all moves
    ref_d = Refinement(sc, sample.select(pool), displacive=[0], fit="none", penalty=100.0, step=0.3,
                       weights={"displace": 1.0}, seed=seed + 2)
    r_d = anneal(ref_d, held, Schedule(rounds=30, sweeps=4, cool=0.9, batch=100_000, batch_sweeps=1,
                                       check_every=10, adapt=0.3), log=None)
    quick(f"displacements only: score {r_d.best['score']:.4f}")
    ref.recompute(track_drift=False)
    ref.step = ref_d.step
result = anneal(ref, held, Schedule(rounds=n_rounds, sweeps=4, cool=0.9,
                                    temperature=0.005 if warm else (0.05 if os.environ.get("DISP_FIRST") else None), batch=100_000, batch_sweeps=1,
                                    check_every=5, adapt=0.3))
quick(f"anneal: score {result.best['score']:.4f}")
for cycle in range(n_reheat):
    # reheat and re-cool; anneal keeps the best held-out state of the cycle
    result = anneal(ref, held, Schedule(rounds=40, sweeps=4, cool=0.9, temperature=T_reheat,
                                        batch=100_000, batch_sweeps=1, check_every=5, adapt=0.3),
                    log=None)
    quick(f"reheat {cycle + 1}: score {result.best['score']:.4f}")

# --- comparison ------------------------------------------------------------
s_ref, u_ref = to_truth(sc)


def table(states, comps):
    v = [truth_correlations.displacement_pair_correlation(graph, s, u, comps) for s, u in states]
    return [np.mean([x[k] for x in v], 0) for k in range(3)]


truth_states = list(zip(truth_sigma, truth_u))
(tn, tj, tms), (rn, rj, rms) = table(truth_states, (0, 0)), table([(s_ref, u_ref)], (0, 0))
(tny, tjy, _), (rny, rjy, _) = table(truth_states, (0, 1)), table([(s_ref, u_ref)], (0, 1))
A, B = 1, 0
print(f"<x^2> x 1e4: truth Zn {tms[0, A] * 1e4:.1f} Cu {tms[0, B] * 1e4:.1f} | "
      f"refined Zn {rms[0, A] * 1e4:.1f} Cu {rms[0, B] * 1e4:.1f}")


def sro(s):
    d = 2 * s.reshape(L, L) - 1.0
    return [np.mean(d * np.roll(d, sh, axis=(0, 1))) for sh in [(1, 0), (1, 1), (2, 0)]]


print("occupancy <s_0 s_lm> (1,0) (1,1) (2,0): truth "
      + " ".join(f"{v:+.3f}" for v in np.mean([sro(s) for s in truth_sigma], 0))
      + " | refined " + " ".join(f"{v:+.3f}" for v in sro(s_ref)))
print("                truth                          refined")
print(" l, m   <xAxB> <xAxA> <xBxB>    []    <xAxB> <xAxA> <xBxB>    []   (normalized; [] = eq. 8 bracket x 1e4)")
for l, mm in [(k, 0) for k in range(1, 9)] + [(0, 1), (1, 1)]:
    row = [tn[A, B, l, mm, 0], tn[A, A, l, mm, 0], tn[B, B, l, mm, 0], tj[l, mm, 0] * 1e4,
           rn[A, B, l, mm, 0], rn[A, A, l, mm, 0], rn[B, B, l, mm, 0], rj[l, mm, 0] * 1e4]
    print(f"{l:2d},{mm:2d}  " + " ".join(f"{v:+6.2f}" for v in row[:4]) + "   "
          + " ".join(f"{v:+6.2f}" for v in row[4:]))
print("<x_0 y_lm>")
for l, mm in [(1, 1), (2, 1), (2, 2), (3, 3)]:
    row = [tny[A, B, l, mm, 0], tny[A, A, l, mm, 0], tny[B, B, l, mm, 0], tjy[l, mm, 0] * 1e4,
           rny[A, B, l, mm, 0], rny[A, A, l, mm, 0], rny[B, B, l, mm, 0], rjy[l, mm, 0] * 1e4]
    print(f"{l:2d},{mm:2d}  " + " ".join(f"{v:+6.2f}" for v in row[:4]) + "   "
          + " ".join(f"{v:+6.2f}" for v in row[4:]))
np.savez(f"butler_roundtrip_seed{seed}.npz", sigma=s_ref, u=u_ref, truth_sigma=truth_sigma,
         truth_u=truth_u, hkl=hkl, y=sample.y, I_truth=I)
print(f"done ({time.time() - t0:.0f} s)")
