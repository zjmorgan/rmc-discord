import numpy as np

from rmc_discord.pairs import Structure
from rmc_discord.pairs.correlations import covariance_from_configurations
from rmc_discord.pairs.design import displacement_design_matrix
from rmc_discord.pairs.symmetry import coefficient_slots, tensor_orbits

from test_size_effect import supercell_pairs


def occupied_structure():
    sites = [
        {
            "label": "A1",
            "species": "C",
            "occupancy": 1.0,
            "x": np.array([0.0, 0.0, 0.0]),
            "U": np.zeros((3, 3)),
        },
        {
            "label": "B1",
            "species": "O",
            "occupancy": 1.0,
            "x": np.array([0.3, 0.1, 0.5]),
            "U": np.zeros((3, 3)),
        },
    ]
    identity = [(np.eye(3), np.zeros(3))]
    return Structure((5.0, 5.5, 4.0, 90, 90, 90), identity, sites)


def displacements(rng, m, L, eps):
    """Correlated Gaussian displacements (Angstrom) of every position."""

    z = rng.normal(size=(m, L, L, L, 2, 3))
    z = z + 0.7 * np.roll(z, 1, axis=3) - 0.4 * np.roll(z, 1, axis=1)
    z[..., 1, :] += 0.5 * z[..., 0, :]
    return eps * z


def exact(structure, sites, u, h):
    m, L = u.shape[0], u.shape[1]
    R = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1)
    pos = R[..., None, :] + sites.x
    I = np.zeros(len(h))
    for q, (hq, Qq) in enumerate(zip(h, structure.Q(h))):
        a = sites.b * np.exp(1j * (u @ Qq))
        a = a - a.mean(axis=(0, 1, 2, 3))
        F = np.einsum("mabcp,abcp->m", a, np.exp(2j * np.pi * (pos @ hq)))
        I[q] = np.mean(np.abs(F) ** 2) / L**3
    return I


def model(structure, sites, u, h):
    m, L = u.shape[0], u.shape[1]
    R, C = covariance_from_configurations(u.reshape(m, L, L, L, 6), 2)
    lookup = {tuple(r): k for k, r in enumerate(R)}
    orbits = supercell_pairs(structure, sites, 2)
    tensors = tensor_orbits(orbits, structure, sites)
    index, value = coefficient_slots(tensors, len(orbits), (3, 3))
    reps = orbits.representatives()
    theta = np.zeros(len(tensors))
    for k, o in enumerate(tensors.orbit):
        p = reps[o]
        i, j, Rp = orbits.i[p], orbits.j[p], orbits.R[p]
        V = C[lookup[tuple(Rp)]][3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
        slot = list(index[p]).index(k)
        theta[k] = np.sum(0.5 * (V + V.T) * value[p, slot])
    A = displacement_design_matrix(
        h, orbits, tensors, sites, structure, thermal=False
    )
    return A @ theta


def direct_sum(structure, sites, u, h):
    """sum over all ordered pairs of b_i b_j Q^T V_ij(R) Q cos(2 pi h.d)."""

    m, L = u.shape[0], u.shape[1]
    R, C = covariance_from_configurations(u.reshape(m, L, L, L, 6), 2)
    Q = structure.Q(h)
    out = np.zeros(len(h))
    for k, r in enumerate(R):
        for i in range(len(sites)):
            for j in range(len(sites)):
                V = C[k][3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
                d = r + sites.x[j] - sites.x[i]
                QVQ = np.einsum("qa,ab,qb->q", Q, V, Q)
                out += (
                    sites.b[i] * sites.b[j] * QVQ * np.cos(2 * np.pi * h @ d)
                )
    return out


def test_tensor_orbits_reproduce_the_direct_sum():
    structure = occupied_structure()
    sites = structure.sites()
    L = 5
    h = np.random.default_rng(0).integers(-12, 13, size=(10, 3)) / L
    u = displacements(np.random.default_rng(1), 30, L, 0.01)
    assert np.allclose(
        model(structure, sites, u, h), direct_sum(structure, sites, u, h)
    )


def test_displacement_correlations_match_supercell_to_second_order():
    structure = occupied_structure()
    sites = structure.sites()
    L = 5
    h = np.random.default_rng(0).integers(-12, 13, size=(10, 3)) / L
    errors = []
    for eps in (0.04, 0.02, 0.01):
        u = displacements(np.random.default_rng(1), 30, L, eps)
        I, M = exact(structure, sites, u, h), model(structure, sites, u, h)
        errors.append(np.abs(I - M).max() / np.abs(M).max())
    # beyond second order only odd-moment sampling noise, of relative
    # order eps, remains (it vanishes in expectation for Gaussian motion)
    assert errors[-1] < 0.02
    assert errors[0] > errors[1] > errors[2]
