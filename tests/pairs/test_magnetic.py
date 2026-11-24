import numpy as np
import pytest

from rmc_discord.crystal.tables import MAGNETIC_LENGTH, magnetic_form_factor
from rmc_discord.pairs import Structure
from rmc_discord.pairs.correlations import covariance_from_configurations
from rmc_discord.pairs.design import magnetic_amplitude, magnetic_design_matrix
from rmc_discord.pairs.symmetry import coefficient_slots, tensor_orbits

from test_size_effect import supercell_pairs

IONS = {"M1": "Fe3+", "M2": "Mn3+"}
MOMENTS = {"Fe3+": 5.92, "Mn3+": 4.90}


def magnetic_structure():
    sites = [
        {"label": "M1", "species": "Fe", "occupancy": 1.0, "x": np.array([0.0, 0.0, 0.0]), "U": np.zeros((3, 3))},
        {"label": "M2", "species": "Mn", "occupancy": 1.0, "x": np.array([0.3, 0.1, 0.5]), "U": np.zeros((3, 3))},
    ]
    return Structure((5.0, 5.5, 4.0, 90, 90, 90), [(np.eye(3), np.zeros(3))], sites)


def spins(rng, m, L):
    """Correlated, anisotropic spin configurations (m, L, L, L, 2, 3)."""
    z = rng.normal(size=(m, L, L, L, 2, 3)) * np.array([1.0, 0.6, 1.4])
    z = z + 0.8 * np.roll(z, 1, axis=2) - 0.5 * np.roll(z, 1, axis=3)
    z[..., 1, :] -= 0.6 * z[..., 0, :]
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def exact(structure, sites, S, h, amplitude):
    """(1/L^3) <|F_perp|^2> of the periodic supercells, mean moment removed."""
    m, L = S.shape[0], S.shape[1]
    R = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1)
    pos = R[..., None, :] + sites.x
    S = S - S.mean(axis=(0, 1, 2, 3))
    Q = structure.Q(h)
    a, _ = amplitude(np.linalg.norm(Q, axis=1))
    I = np.zeros(len(h))
    for q in range(len(h)):
        F = np.einsum("p,mabcpx,abcp->mx", a[q], S, np.exp(2j * np.pi * (pos @ h[q])))
        Qh = Q[q] / np.linalg.norm(Q[q])
        F_perp = F - np.outer(F @ Qh, Qh)
        I[q] = np.mean(np.sum(np.abs(F_perp) ** 2, axis=1)) / L**3
    return I


def theta_from(S, structure, sites, orbits, tensors, half):
    m, L = S.shape[0], S.shape[1]
    R, C = covariance_from_configurations(S.reshape(m, L, L, L, 6), half)
    lookup = {tuple(r): k for k, r in enumerate(R)}
    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    reps = orbits.representatives()
    theta = np.zeros(len(tensors))
    for k, o in enumerate(tensors.orbit):
        p = reps[o]
        i, j, Rp = orbits.i[p], orbits.j[p], orbits.R[p]
        V = C[lookup[tuple(Rp)]][3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
        slot = list(index[p]).index(k)
        theta[k] = np.sum(0.5 * (V + V.T) * value[p, slot])
    return theta


def test_tensor_model_equals_supercell_intensity():
    structure = magnetic_structure()
    sites = structure.sites()
    L, half = 5, 2
    S = spins(np.random.default_rng(3), 20, L)
    h = np.random.default_rng(0).integers(-12, 13, size=(12, 3)) / L
    h = h[np.linalg.norm(h, axis=1) > 0]
    amplitude = magnetic_amplitude(sites, IONS, MOMENTS)
    orbits = supercell_pairs(structure, sites, half)
    tensors = tensor_orbits(orbits, structure, sites)
    theta = theta_from(S, structure, sites, orbits, tensors, half)
    A = magnetic_design_matrix(h, orbits, sites, structure, amplitude, tensors=tensors, thermal=False)
    I = exact(structure, sites, S, h, amplitude)
    assert np.allclose(A @ theta, I, rtol=1e-10, atol=1e-10 * I.max())


def test_isotropic_columns_equal_isotropic_tensors():
    structure = magnetic_structure()
    sites = structure.sites()
    orbits = supercell_pairs(structure, sites, 1)
    tensors = tensor_orbits(orbits, structure, sites)
    amplitude = magnetic_amplitude(sites, IONS, MOMENTS)
    h = np.random.default_rng(1).uniform(-3, 3, (30, 3))
    C = np.random.default_rng(2).uniform(-1, 1, orbits.n_orbits)
    # V_p = C_p I / 3 expressed in the tensor basis
    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    reps = orbits.representatives()
    theta = np.zeros(len(tensors))
    for k, o in enumerate(tensors.orbit):
        p = reps[o]
        slot = list(index[p]).index(k)
        theta[k] = np.sum(C[o] * np.eye(3) / 3 * value[p, slot])
    A_iso = magnetic_design_matrix(h, orbits, sites, structure, amplitude)
    A_ten = magnetic_design_matrix(h, orbits, sites, structure, amplitude, tensors=tensors)
    assert np.allclose(A_iso @ C, A_ten @ theta, rtol=1e-10, atol=1e-12)


def test_mixed_site_amplitudes():
    structure = magnetic_structure()
    sites = structure.sites().select(["M1"])
    amp = magnetic_amplitude(sites, {"M1": [(0.75, "Fe3+"), (0.25, "Mn3+")]}, MOMENTS, g={"Mn3+": 2.0})
    Q = np.array([0.5, 2.0, 5.0])
    a, a2 = amp(Q)
    fe = MAGNETIC_LENGTH * 5.92 * magnetic_form_factor("Fe3+", Q)
    mn = MAGNETIC_LENGTH * 4.90 * magnetic_form_factor("Mn3+", Q)
    assert np.allclose(a[:, 0], 0.75 * fe + 0.25 * mn)
    assert np.allclose(a2[:, 0], 0.75 * fe**2 + 0.25 * mn**2)
    assert np.all(a2[:, 0] >= a[:, 0] ** 2)


def test_self_orbit_gives_paramagnetic_intensity():
    structure = magnetic_structure()
    sites = structure.sites()
    orbits = supercell_pairs(structure, sites, 1)
    amplitude = magnetic_amplitude(sites, IONS, MOMENTS)
    h = np.random.default_rng(4).uniform(-3, 3, (20, 3))
    A = magnetic_design_matrix(h, orbits, sites, structure, amplitude, thermal=False)
    self_orbits = np.unique(orbits.orbit[(orbits.i == orbits.j) & np.all(orbits.R == 0, axis=1)])
    _, a2 = amplitude(np.linalg.norm(structure.Q(h), axis=1))
    assert np.allclose(A[:, self_orbits].sum(1), 2 / 3 * a2.sum(1))
