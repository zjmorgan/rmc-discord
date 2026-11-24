import numpy as np
import pytest

from rmc_discord.pairs.correlations import (
    covariance_from_configurations,
    independent_cluster_covariance,
    translations,
)
from rmc_discord.pairs.rules import find_clusters


def naive_covariance(occupation, R):
    m, L1, L2, L3, n = occupation.shape
    delta = occupation - occupation.mean(axis=(0, 1, 2, 3))
    C = np.zeros((len(R), n, n))
    for k, r in enumerate(R):
        shifted = np.roll(delta, shift=tuple(-r), axis=(1, 2, 3))
        C[k] = np.einsum("mabci,mabcj->ij", delta, shifted) / (
            m * L1 * L2 * L3
        )
    return C


def test_fft_estimator_matches_naive():
    rng = np.random.default_rng(3)
    occupation = rng.random((2, 5, 3, 4, 3)) < 0.4
    R, C = covariance_from_configurations(occupation, (2, 1, 1))
    assert np.allclose(C, naive_covariance(occupation.astype(float), R))


def test_estimator_reversal_symmetry():
    rng = np.random.default_rng(4)
    occupation = rng.random((3, 5, 5, 5, 2)) < 0.3
    R, C = covariance_from_configurations(occupation, 2)
    index = {tuple(r): k for k, r in enumerate(R)}
    for k, r in enumerate(R):
        assert np.allclose(C[k], C[index[tuple(-r)]].T)


def test_max_offset_too_large():
    with pytest.raises(ValueError):
        covariance_from_configurations(np.zeros((1, 4, 4, 4, 1)), 2)


def test_translations():
    R = translations((1, 0, 2))
    assert R.shape == (15, 3)
    assert np.abs(R).max(axis=0).tolist() == [1, 0, 2]


def test_independent_clusters_match_sampling(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    R0, C0 = independent_cluster_covariance(sites, clusters)

    rng = np.random.default_rng(5)
    L, m = 5, 40
    occupation = np.zeros((m, L, L, L, len(sites)))
    for cluster in clusters:
        assert np.all(cluster.offsets == 0)  # fixture pairs share a cell
        p = np.append(sites.occupancy[cluster.members], cluster.vacancy)
        state = rng.choice(len(p), size=(m, L, L, L), p=p)
        for a, i in enumerate(cluster.members):
            occupation[..., i] = state == a
    R, C = covariance_from_configurations(occupation, 1, mean=sites.occupancy)
    zero = np.flatnonzero(np.all(R == 0, axis=1))[0]
    assert np.allclose(C[zero], C0[0], atol=0.02)
    assert np.allclose(np.delete(C, zero, axis=0), 0, atol=0.02)


def test_realizability_bounds_contain_any_configuration(tetragonal):
    from rmc_discord.pairs.correlations import realizability_bounds
    from rmc_discord.pairs.symmetry import orbit_average, pair_orbits

    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    orbits = pair_orbits(tetragonal, sites, 6.0)
    lower, upper = realizability_bounds(orbits, sites, clusters)
    assert np.all(lower <= upper)
    rng = np.random.default_rng(6)
    # perfectly correlated and perfectly anticorrelated extremes stay inside
    occupation = np.zeros((2, 5, 5, 5, len(sites)))
    for cluster in clusters:
        a, b = cluster.members
        z = rng.random((2, 5, 5, 5))
        occupation[..., a] = z < 0.25
        occupation[..., b] = (z >= 0.25) & (z < 0.75)
    R, C = covariance_from_configurations(occupation, 2, mean=sites.occupancy)
    c = orbit_average(orbits, R, C)
    assert np.all(c >= lower - 0.05) and np.all(c <= upper + 0.05)
