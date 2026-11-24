import numpy as np
import pytest

from rmc_discord.pairs.rules import (
    find_clusters,
    find_contacts,
    forbidden_contacts,
)

from conftest import requires_tb5ru6sn18


def test_contacts_are_symmetric(tetragonal):
    sites = tetragonal.sites()
    c = find_contacts(tetragonal, sites, 3.0)
    forward = {(i, j, tuple(R)) for i, j, R in zip(c.i, c.j, c.R)}
    assert all((j, i, tuple(-np.array(R))) in forward for i, j, R in forward)
    dx = sites.x[c.j] + c.R - sites.x[c.i]
    assert np.allclose(tetragonal.distance(dx), c.distance)


def test_split_pair_clusters(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    # A1 (z=0.3) and A2 (z=0.5) are 0.8 A apart: four split pairs
    clusters = find_clusters(tetragonal, sites, 1.0)
    assert sorted(len(c) for c in clusters) == [2, 2, 2, 2]
    assert all(np.isclose(c.vacancy, 0.25) for c in clusters)
    # the split partners are excluded from the forbidden contacts
    fc = forbidden_contacts(tetragonal, sites, clusters, 1.0)
    assert len(fc) == 0


def test_overfull_cluster_raises(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    sites.occupancy[:] = 0.75
    with pytest.raises(ValueError):
        find_clusters(tetragonal, sites, 1.0)


@requires_tb5ru6sn18
def test_tb5ru6sn18_rules(tb5ru6sn18):
    sites = tb5ru6sn18.sites()
    active = sites.select(sites.partial())
    clusters = find_clusters(tb5ru6sn18, active, 1.2)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2] * 32 + [6] * 8
    assert all(c.vacancy == 0 for c in clusters)
    limits = {("Sn", "Sn"): 2.6, ("Sn", "Tb"): 2.8, ("Tb", "Tb"): 3.0}
    fc = forbidden_contacts(tb5ru6sn18, active, clusters, limits)
    assert fc.summary(active) == [
        ("Sn1", "Sn1", 2.44, 3.0),
        ("Sn2", "Tb2", 2.511, 2.0),
        ("Tb2", "Sn2", 2.511, 3.0),
    ]


def test_co_occupied_unit():
    """A partially occupied rigid group (one C with three O at 1.3 A, all
    at occupancy 0.7) is one co-occupied unit, not alternatives."""

    from rmc_discord.pairs import Structure
    from rmc_discord.pairs.correlations import cluster_covariance

    a = 6.0
    r = 1.3 / a
    group = [("C1", "C", (0.5, 0.5, 0.5))] + [
        (f"O{k}", "O", (0.5 + r * np.cos(t), 0.5 + r * np.sin(t), 0.5))
        for k, t in enumerate(np.deg2rad([0, 120, 240]))
    ]
    asymmetric = [
        {
            "label": label,
            "species": species,
            "occupancy": 0.7,
            "x": np.array(x),
            "U": 0.01 * np.eye(3),
        }
        for label, species, x in group
    ]
    identity = [(np.eye(3), np.zeros(3))]
    structure = Structure((a, a, a, 90, 90, 90), identity, asymmetric)
    sites = structure.sites()
    clusters = find_clusters(structure, sites, 1.5)
    assert len(clusters) == 1
    unit = clusters[0]
    assert unit.kind == "co-occupied"
    assert np.isclose(unit.vacancy, 0.3)
    assert np.allclose(cluster_covariance(sites, unit), 0.7 * 0.3)
    # unequal occupancies summing above one fit neither rule
    sites.occupancy[0] = 0.4
    with pytest.raises(ValueError):
        find_clusters(structure, sites, 1.5)
