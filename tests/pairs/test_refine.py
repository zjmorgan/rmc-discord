import numpy as np

from rmc_discord.pairs.correlations import (
    fixed_cluster_orbits,
    independent_cluster_covariance,
)
from rmc_discord.pairs.refine import background_basis, r_factors, refine
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.symmetry import orbit_average, pair_orbits


def problem(rng, m=400, n=12):
    A = rng.normal(size=(m, n)) + 1.0
    c = rng.normal(size=n)
    fixed = (np.array([0, 3]), c[[0, 3]])
    B = background_basis(np.linspace(0.5, 8, m), degree=1)
    beta = np.array([2.0, -0.7])
    return A, c, fixed, B, beta


def test_exact_recovery_with_scale_and_background():
    rng = np.random.default_rng(0)
    A, c, fixed, B, beta = problem(rng)
    y = 2.5 * A @ c + B @ beta
    result = refine(y, np.ones_like(y), A, fixed=fixed, B=B)
    assert np.allclose(result.c, c)
    assert np.isclose(result.scale, 2.5)
    assert np.allclose(result.background, beta)
    assert result.chi2 < 1e-16
    assert np.allclose(result.predict(A, B), y)
    assert not result.free[fixed[0]].any()
    assert np.all(result.esd[fixed[0]] == 0)


def test_esd_matches_scatter():
    rng = np.random.default_rng(1)
    A, c, fixed, B, beta = problem(rng)
    y0 = 2.5 * A @ c + B @ beta
    sigma = np.full_like(y0, 0.5)
    estimates, esds = [], []
    for _ in range(300):
        y = y0 + sigma * rng.normal(size=len(y0))
        result = refine(y, sigma, A, fixed=fixed, B=B)
        estimates.append(result.c)
        esds.append(result.esd)
    scatter = np.std(estimates, axis=0)[result.free]
    predicted = np.mean(esds, axis=0)[result.free]
    assert np.allclose(scatter, predicted, rtol=0.15)
    assert 0.8 < result.reduced_chi2 < 1.2


def test_no_fixed_means_unit_scale_and_rank():
    rng = np.random.default_rng(2)
    A, c, _, _, _ = problem(rng)
    A = np.hstack([A, A[:, :1]])  # duplicate column: rank deficient
    c = np.append(c, 0.0)
    y = A @ c
    result = refine(y, np.ones_like(y), A)
    assert result.scale == 1.0
    assert result.rank == A.shape[1] - 1
    assert np.allclose(result.predict(A), y)


def test_r_factors_zero_for_perfect_model():
    y = np.array([1.0, 2.0, 3.0])
    assert r_factors(y, np.ones(3), y) == (0.0, 0.0)


def test_fixed_cluster_orbits_match_independent_model(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    orbits = pair_orbits(tetragonal, sites, 6.0)
    index, value = fixed_cluster_orbits(orbits, sites, clusters)
    R, C = independent_cluster_covariance(sites, clusters)
    R_all = np.unique(orbits.R, axis=0)
    C_all = np.zeros((len(R_all), len(sites), len(sites)))
    lookup = {tuple(r): k for k, r in enumerate(R_all)}
    for r, block in zip(R, C):
        C_all[lookup[tuple(r)]] = block
    c = orbit_average(orbits, R_all, C_all)
    assert np.allclose(c[index], value)
    others = np.setdiff1d(np.arange(orbits.n_orbits), index)
    assert np.allclose(c[others], 0)
    # self pairs (x - x^2) and the split partners (-x_a x_b) are fixed
    assert np.isclose(value[0], 0.25 - 0.25**2) or np.isclose(
        value[0], 0.5 - 0.5**2
    )


def test_radial_background_basis():
    from rmc_discord.pairs.refine import radial_background_basis

    knots = np.linspace(0, 10, 11)
    Q = np.random.default_rng(0).random(500) * 10
    B = radial_background_basis(Q, knots)
    assert np.allclose(B.sum(axis=1), 1)
    values = np.sin(knots) + knots
    assert np.allclose(B @ values, np.interp(Q, knots, values))
