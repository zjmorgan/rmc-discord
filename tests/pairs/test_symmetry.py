import numpy as np
import pytest

from rmc_discord.pairs.correlations import independent_cluster_covariance
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.design import design_matrix, evaluate
from rmc_discord.pairs.symmetry import (
    orbit_average,
    orbit_pair_terms,
    pair_orbits,
    position_permutations,
)

from conftest import requires_tb5ru6sn18


def test_permutations_are_bijections(tetragonal):
    sites = tetragonal.sites()
    perm, shift = position_permutations(tetragonal, sites)
    for g, (W, t) in enumerate(tetragonal.operators):
        assert sorted(perm[g]) == list(range(len(sites)))
        image = sites.x @ W.T + t
        assert np.allclose(image, sites.x[perm[g]] + shift[g])


def test_orbits_are_closed_and_isometric(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    orbits = pair_orbits(tetragonal, sites, 6.0)
    dist = tetragonal.distance(orbits.d)
    assert np.allclose(dist, orbits.distance[orbits.orbit])
    assert np.all(np.diff(orbits.distance) >= -1e-9)
    # reversal partner of every pair lies in the same orbit
    key = {
        (i, j, tuple(R)): o
        for i, j, R, o in zip(orbits.i, orbits.j, orbits.R, orbits.orbit)
    }
    for i, j, R, o in zip(orbits.i, orbits.j, orbits.R, orbits.orbit):
        assert key[(j, i, tuple(-R))] == o
    # P4 with 4-fold copies: no orbit can exceed 2 * 4 pairs per (i-class)
    counts = np.bincount(orbits.orbit)
    assert counts.max() <= 2 * 4 * 2


def test_symmetric_covariance_is_constant_on_orbits(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    R, C = independent_cluster_covariance(sites, clusters)
    orbits = pair_orbits(tetragonal, sites, 1.0)
    c = orbit_average(orbits, np.vstack([R]), C)
    values = C[
        [{tuple(r): k for k, r in enumerate(R)}[tuple(r)] for r in orbits.R],
        orbits.i,
        orbits.j,
    ]
    assert np.allclose(values, c[orbits.orbit])


def test_design_matrix_matches_evaluate(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    orbits = pair_orbits(tetragonal, sites, 7.0)
    rng = np.random.default_rng(2)
    c = rng.normal(size=orbits.n_orbits)
    h = rng.normal(size=(30, 3)) * 3
    widths = np.array([0.2, 0.25, 0.3])
    A = design_matrix(
        h, orbits, sites, tetragonal, widths=widths, max_elements=3000
    )
    terms = orbit_pair_terms(orbits, sites, c).bin_average(widths)
    assert np.allclose(A @ c, evaluate(h, terms, tetragonal, check=True))


@requires_tb5ru6sn18
def test_tb5ru6sn18_orbits(tb5ru6sn18):
    sites = tb5ru6sn18.sites()
    active = sites.select(sites.partial())
    orbits = pair_orbits(tb5ru6sn18, active, 8.0)
    assert orbits.distance[0] == 0
    # self pairs: one orbit per asymmetric label
    self_orbits = np.unique(orbits.orbit[orbits.distance[orbits.orbit] == 0])
    assert len(self_orbits) == 3
    # every orbit's pair count divides the group-and-reversal size
    assert np.all((2 * 192) % np.bincount(orbits.orbit) == 0)


def test_cylinder_extends_pairs_along_unique_axis(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    sphere = pair_orbits(tetragonal, sites, 6.0)
    tube = pair_orbits(tetragonal, sites, 6.0, cylinder=((0, 0, 1), 1.0, 20.0))
    assert len(tube) > len(sphere)
    r = tube.d @ tetragonal.A.T
    long = np.linalg.norm(r, axis=1) > 6.0 + 1e-6
    assert np.all(np.hypot(r[long, 0], r[long, 1]) <= 1.0 + 1e-6)
    assert np.abs(r[long, 2]).max() > 15
    # a cylinder along a direction the 4-fold axis does not preserve
    with pytest.raises(ValueError):
        pair_orbits(tetragonal, sites, 3.0, cylinder=((1, 0, 0), 0.5, 20.0))
