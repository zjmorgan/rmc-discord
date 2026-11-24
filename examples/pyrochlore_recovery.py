"""
Known-answer test: refine pair correlations of the synthetic Ho2Ti2O7 ice data.

The data (``make_pyrochlore_data.py``) are ensemble averages of a
nearest-neighbor ice model on the Ho pyrochlore sublattice, so the true
correlations are known from the configurations in ``truth.npz``:

* ``occ_ice``: occupational ice (two of four Ho per tetrahedron present);
  occupancy covariances, absolute scale fixed by the self term c(1 - c).
* ``spin_ice``: Ising moments along the local <111> axes; spin
  correlations ⟨S_i S_j^T⟩ are anisotropic, so the isotropic model
  ⟨S_i·S_j⟩ is expected to fail where the symmetric-tensor model works.
  The unit-spin trace of the self tensors fixes the absolute scale; the
  anisotropic part of the self tensor is not identifiable in a cubic
  crystal (the local axes average to an isotropic sum).

Ice correlations decay as a power law (pinch points), so the refined
range r_max is scanned to show the effect of the finite cutoff.

The synthetic intensities are point values at the bin centers (not bin
integrals), so the model is evaluated without bin averaging.

Usage: python pyrochlore_recovery.py [r_max ...]   (default 8 12)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from rmc_discord.crystal.structure import Crystal
from rmc_discord.data import volume
from rmc_discord.pairs import Structure
from rmc_discord.pairs.constraints import affine_parameterization, self_trace_rows
from rmc_discord.pairs.correlations import covariance_from_configurations, fixed_cluster_orbits
from rmc_discord.pairs.design import design_matrix, magnetic_amplitude, magnetic_design_matrix
from rmc_discord.pairs.normal import accumulate, held_out_chi2, refine_normal
from rmc_discord.pairs.refine import background_basis
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.sampling import from_volume
from rmc_discord.pairs.symmetry import coefficient_slots, orbit_average, pair_orbits, tensor_orbits

root = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
root = root / "pyrochlore_synthetic"
r_values = [float(v) for v in sys.argv[1:]] or [8.0, 12.0]
a, L = 10.1, 16
truth = np.load(root / "truth.npz")


def truth_fields(active):
    """Map the truth atoms onto (cell, active position) of the refined structure."""
    frac = truth["r"] / a
    cell = np.floor(frac + 1e-6).astype(int)
    x = frac - cell
    d = (x[:, None, :] - active.x[None] + 0.5) % 1 - 0.5
    pos = np.argmin(np.abs(d).max(axis=2), axis=1)
    assert np.abs(d[np.arange(len(pos)), pos]).max() < 1e-4
    return np.mod(cell, L), pos


def lattice_array(values, cell, pos, n):
    """(m, n_atoms, ...) values to (m, L, L, L, n, ...)."""
    out = np.zeros((values.shape[0], L, L, L, n) + values.shape[2:])
    out[:, cell[:, 0], cell[:, 1], cell[:, 2], pos] = values
    return out


def truth_chi2(train, test, c):
    """Held-out chi2 of fixed correlations c with only scale and background refined."""
    nd, nb = train.n_design, train.n_background
    M = np.zeros((nd + nb, 1 + nb))
    M[:nd, 0] = c
    M[nd:, 1:] = np.eye(nb)
    p = np.linalg.lstsq(M.T @ train.G @ M, M.T @ train.g, rcond=None)[0]
    return test.chi2(M @ p) / test.n_obs


def background(structure):
    return lambda hs: background_basis(np.linalg.norm(structure.Q(hs), axis=1), degree=1)


def occupational(r_max):
    structure = Structure.from_crystal(Crystal.from_cif(str(root / "Ho2Ti2O7_half.cif")))
    active = structure.sites().select(["Ho1"])
    obs = from_volume(volume.load(root / "occ_ice.nxs"), structure, bragg_radius=0.0, held_out=0.2)
    orbits = pair_orbits(structure, active, r_max)
    fixed = fixed_cluster_orbits(orbits, active, find_clusters(structure, active, 0.5))
    cell, pos = truth_fields(active)
    occ = lattice_array(truth["sigma"].astype(float)[..., None], cell, pos, len(active))[..., 0]
    R, C = covariance_from_configurations(occ, 7)
    c_true = orbit_average(orbits, R, C)

    def design(hs):
        return design_matrix(hs, orbits, active, structure, widths=None, projection=obs.projection,
                             thermal=False)

    t = time.time()
    train, test = accumulate(design, obs.h, obs.y, obs.sigma, background(structure), split=obs.held)
    result = refine_normal(train, fixed=fixed)
    print(f"\noccupational ice, r_max {r_max:g} A: {orbits.n_orbits} orbits, {len(obs)} observations,"
          f" held-out chi2 {held_out_chi2(result, test):.3f} (truth truncated: {truth_chi2(train, test, c_true):.3f};"
          f" {time.time() - t:.1f} s)")
    print(f"  {'r (A)':>7s} {'refined':>9s} {'esd':>8s} {'truth':>9s}")
    for o in range(min(orbits.n_orbits, 10)):
        print(f"  {orbits.distance[o]:7.3f} {result.c[o]:9.5f} {result.esd[o]:8.5f} {c_true[o]:9.5f}")


def magnetic(r_max):
    structure = Structure.from_crystal(Crystal.from_cif(str(root / "Ho2Ti2O7.cif")))
    active = structure.sites().select(["Ho1"])
    obs = from_volume(volume.load(root / "spin_ice.nxs"), structure, bragg_radius=0.0, held_out=0.2)
    orbits = pair_orbits(structure, active, r_max)
    amplitude = magnetic_amplitude(active, {"Ho1": "Ho3+"}, {"Ho3+": 10.0})
    cell, pos = truth_fields(active)
    S = (2 * truth["sigma"] - 1)[..., None] * truth["axis"][None]
    S = lattice_array(S, cell, pos, len(active))
    R, C = covariance_from_configurations(S.reshape(*S.shape[:4], -1), 7)
    self_orbits = np.unique(orbits.orbit[orbits.distance[orbits.orbit] < 1e-8])

    # isotropic: <S_i . S_j> per orbit, self = 1 sets the scale
    trace = np.einsum("kiaja->kij", C.reshape(len(C), len(active), 3, len(active), 3))
    c_iso = orbit_average(orbits, R, trace)
    design = lambda hs: magnetic_design_matrix(hs, orbits, active, structure, amplitude, widths=None,
                                               projection=obs.projection, thermal=False)
    t = time.time()
    train, test = accumulate(design, obs.h, obs.y, obs.sigma, background(structure), split=obs.held)
    iso = refine_normal(train, fixed=(self_orbits, np.ones(len(self_orbits))))
    print(f"\nspin ice, r_max {r_max:g} A: {orbits.n_orbits} orbits, {len(obs)} observations")
    print(f"  isotropic <S.S>: held-out chi2 {held_out_chi2(iso, test):.3f}"
          f" (truth truncated: {truth_chi2(train, test, c_iso):.3f}; {time.time() - t:.1f} s)")

    # symmetric tensor <S_i S_j^T>
    tensors = tensor_orbits(orbits, structure, active)
    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    reps = orbits.representatives()
    lookup = {tuple(r): k for k, r in enumerate(R)}
    theta_true = np.zeros(len(tensors))
    for k, o in enumerate(tensors.orbit):
        p = reps[o]
        i, j = orbits.i[p], orbits.j[p]
        V = C[lookup[tuple(orbits.R[p])]][3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
        theta_true[k] = np.sum(0.5 * (V + V.T) * value[p, list(index[p]).index(k)])
    design_t = lambda hs: magnetic_design_matrix(hs, orbits, active, structure, amplitude, tensors=tensors,
                                                 widths=None, projection=obs.projection, thermal=False)
    t = time.time()
    train, test = accumulate(design_t, obs.h, obs.y, obs.sigma, background(structure), split=obs.held)
    ten = refine_normal(train, parameterization=affine_parameterization(len(tensors), *self_trace_rows(orbits, tensors)))
    print(f"  tensor <S S^T>: {len(tensors)} coefficients, held-out chi2 {held_out_chi2(ten, test):.3f}"
          f" (truth truncated: {truth_chi2(train, test, theta_true):.3f})"
          f" ({time.time() - t:.1f} s)")
    print(f"  tensor coefficients (unit-spin trace fixed): {'r (A)':>7s} {'refined':>9s} {'esd':>8s} {'truth':>9s}")
    for k in np.flatnonzero(tensors.orbit < 3)[:12]:
        print(f"  {'':45s}{orbits.distance[tensors.orbit[k]]:7.3f} {ten.c[k]:9.4f} {ten.esd[k]:8.4f}"
              f" {theta_true[k]:9.4f}")
    print(f"  {'r (A)':>7s} {'iso refined':>11s} {'iso truth':>9s}")
    for o in range(min(orbits.n_orbits, 6)):
        print(f"  {orbits.distance[o]:7.3f} {iso.c[o]:11.4f} {c_iso[o]:9.4f}")


for r in r_values:
    occupational(r)
    magnetic(r)
