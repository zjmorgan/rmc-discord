import numpy as np
import pytest

from rmc_discord.pairs.constraints import (
    homometric_groups,
    parameterize,
    sum_rule_rows,
    tie_rows,
)
from rmc_discord.pairs.correlations import (
    covariance_from_configurations,
    fixed_cluster_orbits,
)
from rmc_discord.pairs.refine import refine
from rmc_discord.pairs.rules import find_clusters
from rmc_discord.pairs.design import design_matrix
from rmc_discord.pairs.symmetry import orbit_average, pair_orbits

from conftest import requires_tb5ru6sn18


@pytest.fixture
def complete_pairs(tetragonal):
    """Split pairs A1/A2 with occupancies summing to one (no vacancy)."""

    sites = tetragonal.sites().select(["A1", "A2"])
    sites.occupancy[:] = 0.5
    clusters = find_clusters(tetragonal, sites, 1.0)
    assert all(c.vacancy == 0 for c in clusters)
    return tetragonal, sites, clusters


def single_occupancy_configurations(rng, sites, clusters, m=6, L=7):
    """Correlated configurations with exactly one atom per cluster."""

    z = rng.normal(size=(m, L, L, L, len(clusters)))
    z = z + 0.8 * np.roll(z, 1, axis=1) - 0.5 * np.roll(z, 1, axis=3)
    occupation = np.zeros((m, L, L, L, len(sites)))
    for k, cluster in enumerate(clusters):
        a, b = cluster.members
        occupation[..., a] = z[..., k] > 0
        occupation[..., b] = z[..., k] <= 0
    return occupation


def test_sum_rules_hold_for_single_occupancy(complete_pairs):
    structure, sites, clusters = complete_pairs
    orbits = pair_orbits(structure, sites, 9.0)
    rows = sum_rule_rows(orbits, clusters)
    assert len(rows) > 0
    rng = np.random.default_rng(0)
    occupation = single_occupancy_configurations(rng, sites, clusters)
    R, C = covariance_from_configurations(occupation, 3)
    c = orbit_average(orbits, R, C)
    assert np.allclose(rows @ c, 0, atol=1e-12)
    # a generic covariance violates them
    assert not np.allclose(rows @ rng.normal(size=len(c)), 0)


def test_no_rules_with_vacancies(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    orbits = pair_orbits(tetragonal, sites, 6.0)
    assert len(sum_rule_rows(orbits, clusters)) == 0


def test_parameterization_satisfies_everything(complete_pairs):
    structure, sites, clusters = complete_pairs
    orbits = pair_orbits(structure, sites, 9.0)
    fixed = fixed_cluster_orbits(orbits, sites, clusters)
    group = homometric_groups(orbits, sites)
    rows = np.vstack(
        [sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])]
    )
    p = parameterize(orbits.n_orbits, fixed, rows)
    assert p.n_free < orbits.n_orbits - len(fixed[0])
    rng = np.random.default_rng(1)
    for _ in range(5):
        c = p.coefficients(rng.normal(size=p.n_free))
        assert np.allclose(rows @ c, 0, atol=1e-10)
        assert np.allclose(c[fixed[0]], fixed[1])


def test_refine_recovers_constrained_truth(complete_pairs):
    structure, sites, clusters = complete_pairs
    orbits = pair_orbits(structure, sites, 9.0)
    fixed = fixed_cluster_orbits(orbits, sites, clusters)
    group = homometric_groups(orbits, sites)
    rows = np.vstack(
        [sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])]
    )
    p = parameterize(orbits.n_orbits, fixed, rows)
    rng = np.random.default_rng(2)
    truth = p.coefficients(0.05 * rng.normal(size=p.n_free))
    h = rng.normal(size=(3 * orbits.n_orbits, 3)) * 2.5
    A = design_matrix(h, orbits, sites, structure)
    y = 1.7 * A @ truth
    r = refine(y, np.ones_like(y), A, parameterization=p)
    assert np.isclose(r.scale, 1.7)
    assert np.allclose(r.c, truth, atol=1e-8)
    assert r.rank == r.n_params
    assert np.all(r.esd[fixed[0]] < 1e-12)


@requires_tb5ru6sn18
def test_tb5ru6sn18_constraints(tb5ru6sn18):
    sites = tb5ru6sn18.sites()
    active = sites.select(sites.partial())
    clusters = find_clusters(tb5ru6sn18, active, 1.2)
    orbits = pair_orbits(tb5ru6sn18, active, 14.0)
    group = homometric_groups(orbits, active)
    tied = [o for o in range(orbits.n_orbits) if group[o] != o]
    assert len(tied) >= 1
    for o in tied:
        assert np.isclose(orbits.distance[o], orbits.distance[group[o]])
    fixed = fixed_cluster_orbits(orbits, active, clusters)
    rows = np.vstack(
        [sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])]
    )
    p = parameterize(orbits.n_orbits, fixed, rows)
    # Sn2 occupancy 0.1667 (not 1/6) leaves a tiny violation only
    assert p.residual < 1e-3
    assert p.n_free < (orbits.n_orbits - len(fixed[0])) // 2


def test_participation_scan_finds_correlated_unit(complete_pairs):
    """Correlated split pairs: the model with the pairs participating
    must beat the uncorrelated null model on held-out positions."""

    from rmc_discord.pairs.participation import participation_scan

    structure, sites, clusters = complete_pairs
    orbits = pair_orbits(structure, sites, 9.0)
    fixed = fixed_cluster_orbits(orbits, sites, clusters)
    group = homometric_groups(orbits, sites)
    rows = np.vstack(
        [sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])]
    )
    p = parameterize(orbits.n_orbits, fixed, rows)
    rng = np.random.default_rng(3)
    truth = p.coefficients(0.05 * rng.normal(size=p.n_free))
    h = rng.normal(size=(4000, 3)) * 2.5
    A = design_matrix(h, orbits, sites, structure)
    y0 = A @ truth
    sigma = 0.02 * np.abs(y0).max() * np.ones_like(y0)
    y = y0 + sigma * rng.normal(size=len(y0))
    train = rng.random(len(y)) < 0.5
    units = {"pairs": ["A1", "A2"]}
    models = participation_scan(
        y, sigma, A, orbits, sites, units, fixed, rows, train
    )
    assert models[0].units == ("pairs",)
    assert 0.8 < models[0].chi2_test < 1.2
    assert models[-1].chi2_test > 2


def test_unit_rules_hold_for_rigid_groups():
    """Configurations where each CO3-like group is wholly present or
    absent satisfy the unit equalities exactly."""

    from rmc_discord.pairs import Structure
    from rmc_discord.pairs.constraints import unit_rows

    a = 6.0
    r = 1.3 / a
    group = [("C1", "C", (0.5, 0.5, 0.5))] + [
        (f"O{k}", "O", (0.5 + r * np.cos(t), 0.5 + r * np.sin(t), 0.5))
        for k, t in enumerate(np.deg2rad([0, 120, 240]))
    ]
    asymmetric = [
        {
            "label": l,
            "species": s,
            "occupancy": 0.7,
            "x": np.array(x),
            "U": 0.01 * np.eye(3),
        }
        for l, s, x in group
    ]
    structure = Structure(
        (a, a, a, 90, 90, 90), [(np.eye(3), np.zeros(3))], asymmetric
    )
    sites = structure.sites()
    clusters = find_clusters(structure, sites, 1.5)
    orbits = pair_orbits(structure, sites, 12.5)
    rows = unit_rows(orbits, clusters)
    assert len(rows) > 0
    rng = np.random.default_rng(5)
    z = rng.normal(size=(4, 5, 5, 5))
    present = (z + 0.7 * np.roll(z, 1, axis=3)) > -0.5
    occupation = np.repeat(present[..., None], len(sites), axis=-1)
    R, C = covariance_from_configurations(occupation, 2)
    c = orbit_average(orbits, R, C)
    assert np.allclose(rows @ c, 0, atol=1e-12)
    fixed = fixed_cluster_orbits(orbits, sites, clusters)
    assert np.allclose(fixed[1], 0.7 * 0.3)
