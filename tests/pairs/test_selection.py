import itertools

import numpy as np
import pytest

from rmc_discord.pairs import Structure
from rmc_discord.pairs.selection import (
    asymmetric_unit,
    bragg_mask,
    canonical,
    is_allowed,
    laue_operators,
    line_mask,
    satellite_mask,
)


@pytest.fixture
def fcc():
    """Synthetic Fm-3m operator set: 48 rotations x 4 centerings."""

    rotations = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((-1, 1), repeat=3):
            W = np.zeros((3, 3))
            W[np.arange(3), p] = s
            rotations.append(W)
    centring = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    ops = [(W, np.array(t, float)) for W in rotations for t in centring]
    return Structure((10, 10, 10, 90, 90, 90), ops, [])


def test_reflection_conditions(fcc):
    h = np.array([[1, 1, 1], [2, 0, 0], [1, 0, 0], [0.5, 0, 0], [2, 1, 1]])
    assert is_allowed(h, fcc).tolist() == [True, True, False, False, False]


def test_laue_group_size(fcc):
    assert len(laue_operators(fcc)) == 48


def test_bragg_mask(fcc):
    r = 0.06  # inverse Angstrom; 0.1 r.l.u. = 0.0628 here
    h = np.array([[2.0, 0, 0.09], [2.0, 0, 0.11], [1.0, 0, 0]])
    assert bragg_mask(h, fcc, r).tolist() == [True, False, False]


def test_satellites_of_x_point(fcc):
    r = 0.03
    h = np.array([[1, 0, 0], [0, 3, 0], [1, 1, 0], [1, 1, 1], [0.5, 0, 0]])
    mask = satellite_mask(h, fcc, (1, 0, 0), r)
    assert mask.tolist() == [True, True, True, False, False]


def test_lines_through_x_points(fcc):
    r = 0.03
    h = np.array(
        [
            [1, 0.37, 0],  # on (1, t, 0)
            [1, 7.6, 0],  # far along the same line
            [2, 3.3, 1],  # (0, t, 1) + (2, 2, 0)
            [1, 0.37, 0.5],  # half a r.l.u. off the line
            [0.5, 0.5, 0.5],
        ]
    )
    mask = line_mask(h, fcc, (1, 0, 0), (0, 1, 0), r)
    assert mask.tolist() == [True, True, True, False, False]


def test_asymmetric_unit_counts_orbits(fcc):
    n = 3
    g = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    index = tuple((grid + n).T)
    keep = asymmetric_unit(index, (2 * n + 1,) * 3, (n, n, n), fcc)
    W = laue_operators(fcc)
    orbits = {frozenset(map(tuple, (p @ W).tolist())) for p in grid}
    assert keep.sum() == len(orbits)
    reps = grid[keep]
    assert len({frozenset(map(tuple, (p @ W).tolist())) for p in reps}) == len(
        orbits
    )


def test_canonical_is_shared_by_equivalents(fcc):
    rng = np.random.default_rng(0)
    h = np.round(rng.normal(size=(20, 3)) * 4, 2)
    rep = canonical(h, fcc)
    for W in laue_operators(fcc)[::7]:
        assert np.allclose(canonical(h @ W, fcc), rep)


@pytest.fixture
def p_6():
    """Space group P-6 (Laue class 6/m) in the hexagonal setting."""

    from rmc_discord.pairs.structure import parse_symop

    ops = [
        "x, y, z",
        "-y, x-y, z",
        "-x+y, -x, z",
        "x, y, -z",
        "-y, x-y, -z",
        "-x+y, -x, -z",
    ]
    return Structure(
        (9.7, 9.7, 9.6, 90, 90, 120), [parse_symop(o) for o in ops], []
    )


def test_extra_operators_close_the_laue_group(p_6):
    from rmc_discord.pairs.selection import coset_operators

    assert len(laue_operators(p_6)) == 12  # 6/m
    assert len(laue_operators(p_6, ["y, x, z"])) == 24  # 6/mmm
    W = coset_operators(p_6, ["y, x, z"])
    assert len(W) == 2 and np.array_equal(W[0], np.eye(3))


def test_fundamental_domain_counts_orbits(p_6):
    from rmc_discord.pairs.selection import in_fundamental_domain

    g = np.arange(-3, 4)
    h = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    extra = ["y, x, z"]
    keep = in_fundamental_domain(h, p_6, extra)
    W = laue_operators(p_6, extra)
    # a cube of indices is not closed under hexagonal rotations: count
    # the orbits whose canonical representative lies inside it
    grid = set(map(tuple, h.tolist()))
    reps = {tuple(r) for r in canonical(h.astype(float), p_6, extra=extra)}
    expected = {r for r in reps if r in grid}
    kept = h[keep].astype(int)
    assert keep.sum() == len(expected)
    kept_orbits = {frozenset(map(tuple, (p @ W).tolist())) for p in kept}
    assert len(kept_orbits) == len(kept)


def test_twin_averaged_design_matrix(p_6):
    """Averaging over coset representatives makes the model invariant
    under the extra operator."""

    from rmc_discord.pairs import Structure
    from rmc_discord.pairs.design import design_matrix
    from rmc_discord.pairs.selection import coset_operators
    from rmc_discord.pairs.structure import parse_symop
    from rmc_discord.pairs.symmetry import pair_orbits

    ops = [W for W, _ in p_6.operators]
    site = {
        "label": "X1",
        "species": "C",
        "occupancy": 0.5,
        "x": np.array([0.31, 0.12, 0.2]),
        "U": 0.01 * np.eye(3),
    }
    structure = Structure(p_6.cell, p_6.operators, [site])
    sites = structure.sites()
    orbits = pair_orbits(structure, sites, 6.0)
    rng = np.random.default_rng(0)
    h = rng.normal(size=(10, 3)) * 2
    c = rng.normal(size=orbits.n_orbits)
    W = coset_operators(structure, ["y, x, z"])
    A = design_matrix(h, orbits, sites, structure, operators=W)
    mirror = parse_symop("y, x, z")[0].astype(int)
    A_m = design_matrix(h @ mirror, orbits, sites, structure, operators=W)
    assert np.allclose(A @ c, A_m @ c)
    plain = design_matrix(h, orbits, sites, structure)
    plain_m = design_matrix(h @ mirror, orbits, sites, structure)
    assert not np.allclose(plain @ c, plain_m @ c)


def test_powder_ring_mask_finds_rings():
    from rmc_discord.pairs.selection import powder_ring_mask

    rng = np.random.default_rng(0)
    Q = rng.random(200000) * 10
    y = 1 + 0.1 * Q + rng.normal(0, 0.05, len(Q))
    for q0 in (2.5, 6.1):
        y += 3 * np.exp(-0.5 * ((Q - q0) / 0.01) ** 2)
    mask, rings = powder_ring_mask(Q, y)
    assert np.all(np.abs(Q[mask][:, None] - [2.5, 6.1]).min(axis=1) < 0.05)
    assert mask[np.abs(Q - 2.5) < 0.005].all()
    assert mask[np.abs(Q - 6.1) < 0.005].all()
    assert mask.mean() < 0.02


def test_with_occupancies(tetragonal):
    s = tetragonal.with_occupancies({"A1": 0.4})
    sites = s.sites()
    assert np.allclose(sites.occupancy[sites.label == "A1"], 0.4)
    assert np.allclose(sites.occupancy[sites.label == "A2"], 0.5)
    with pytest.raises(KeyError):
        tetragonal.with_occupancies({"Z9": 0.1})
