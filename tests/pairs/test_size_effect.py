import numpy as np

from rmc_discord.pairs import (
    Structure,
    evaluate,
    pair_terms_from_covariance,
)
from rmc_discord.pairs.correlations import covariance_from_configurations
from rmc_discord.pairs.design import size_effect_design_matrix
from rmc_discord.pairs.symmetry import (
    PairOrbits,
    vector_orbits,
    vector_slots,
)


def chain_structure():
    """P1 cell with two partially occupied positions, short c axis."""

    sites = [
        {
            "label": "C1",
            "species": "C",
            "occupancy": 0.7,
            "x": np.array([0.0, 0.0, 0.0]),
            "U": np.zeros((3, 3)),
        },
        {
            "label": "O1",
            "species": "O",
            "occupancy": 0.6,
            "x": np.array([0.3, 0.1, 0.5]),
            "U": np.zeros((3, 3)),
        },
    ]
    return Structure(
        (5.0, 5.5, 4.0, 90, 90, 90), [(np.eye(3), np.zeros(3))], sites
    )


def configurations(rng, m, L, x, eps):
    """Independent occupations; present atoms shift along c by
    eps * (n(R + c) - n(R - c)) in Angstrom (toward a vacancy when
    eps < 0)."""

    n = (rng.random((m, L, L, L, 2)) < x).astype(float)
    shift = np.roll(n, -1, axis=3) - np.roll(n, 1, axis=3)
    v = np.zeros(n.shape + (3,))
    v[..., 2] = eps * n * shift
    return n, v


def brute_force(structure, sites, n, v, h):
    m, L = n.shape[0], n.shape[1]
    R = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1)
    pos = R[..., None, :] + sites.x
    Q = structure.Q(h)
    I = np.zeros(len(h))
    for q, (hq, Qq) in enumerate(zip(h, Q)):
        a = sites.b * n * np.exp(1j * (v @ Qq))
        # fluctuation about the global mean amplitude of each position,
        # the same mean the covariance estimators subtract
        a = a - a.mean(axis=(0, 1, 2, 3))
        F = np.einsum("mabcp,abcp->m", a, np.exp(2j * np.pi * (pos @ hq)))
        I[q] = np.mean(np.abs(F) ** 2) / L**3
    return I


def supercell_pairs(structure, sites, half):
    """Every pair with |R_a| <= half (P1: orbit = pair and its reverse),
    i.e. all separations of a periodic (2 half + 1)^3 supercell."""

    g = np.arange(-half, half + 1)
    T = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n = len(sites)
    i, j, k = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(len(T)), indexing="ij"
    )
    i, j, R = i.ravel(), j.ravel(), T[k.ravel()]
    key = {(a, b, tuple(r)): q for q, (a, b, r) in enumerate(zip(i, j, R))}
    canon = np.array(
        [
            min(q, key[(b, a, tuple(-r))])
            for q, (a, b, r) in enumerate(zip(i, j, R))
        ]
    )
    _, orbit = np.unique(canon, return_inverse=True)
    d = R + sites.x[j] - sites.x[i]
    dist = structure.distance(d)
    distance = np.array([dist[orbit == o][0] for o in range(orbit.max() + 1)])
    return PairOrbits(i, j, R, d, orbit.ravel(), distance)


def model(structure, sites, n, v, h):
    m, L = n.shape[0], n.shape[1]
    mean_n = n.mean(axis=(0, 1, 2, 3))
    R, C = covariance_from_configurations(n, 2, mean=mean_n)
    occ = evaluate(
        h, pair_terms_from_covariance(sites, R, C, thermal=False), structure
    )
    # size-effect vectors from the joint covariance of n and v
    joint = np.concatenate([n, v.reshape(m, L, L, L, -1)], axis=-1)
    mean = np.concatenate([mean_n, np.zeros(6)])
    Rj, Cj = covariance_from_configurations(joint, 2, mean=mean)
    lookup = {tuple(r): k for k, r in enumerate(Rj)}
    orbits = supercell_pairs(structure, sites, 2)
    vectors = vector_orbits(orbits, structure, sites)
    index, value = vector_slots(vectors, len(orbits))
    theta = np.zeros(len(vectors))
    reps = orbits.representatives()
    for k, o in enumerate(vectors.orbit):
        p = reps[o]
        i, j, Rp = orbits.i[p], orbits.j[p], orbits.R[p]
        block = Cj[lookup[tuple(Rp)]]
        sigma = (
            block[i, 2 + 3 * j : 5 + 3 * j] - block[2 + 3 * i : 5 + 3 * i, j]
        )
        slot = list(index[p]).index(k)
        theta[k] = sigma @ value[p, slot]
    A = size_effect_design_matrix(
        h, orbits, vectors, sites, structure, thermal=False
    )
    return occ, A @ theta


def test_size_effect_matches_supercell_to_first_order():
    structure = chain_structure()
    sites = structure.sites()
    L = 5
    rng = np.random.default_rng(0)
    h = rng.integers(-12, 13, size=(12, 3)) / L
    h[:, 2] += 1.0  # nonzero Q_z, where motion along c is visible
    errors = []
    for eps in (-0.04, -0.02, -0.01):
        n, v = configurations(np.random.default_rng(1), 40, L, 0.65, eps)
        exact = brute_force(structure, sites, n, v, h)
        occ, size = model(structure, sites, n, v, h)
        first_order = exact - occ
        assert np.abs(size).max() > 0
        errors.append(np.abs(first_order - size).max() / np.abs(size).max())
    # the remaining difference is second order in the displacement
    assert errors[-1] < 0.05
    assert errors[0] / errors[-1] > 3
    # the sign matters: the opposite sign is far off
    assert np.abs(first_order + size).max() > np.abs(size).max()
