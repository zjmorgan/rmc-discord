import numpy as np
import pytest

from rmc_discord.pairs import Structure
from rmc_discord.pairs.bragg import (
    Profile,
    bragg_design,
    fit_profile,
    structure_factor,
)


@pytest.fixture
def simple():
    site = {
        "label": "X1",
        "species": "C",
        "occupancy": 0.8,
        "x": np.array([0.1, 0.2, 0.3]),
        "U": 0.005 * np.eye(3),
    }
    identity = [(np.eye(3), np.zeros(3))]
    return Structure((5.0, 5.0, 6.0, 90, 90, 120), identity, [site])


def test_structure_factor_of_one_atom(simple):
    h = np.array([[1, 0, 0], [1, -1, 2]], float)
    sites = simple.sites()
    Q = simple.Q(h)
    W = 0.5 * 0.005 * np.sum(Q**2, axis=1)
    expected = (
        sites.b[0] * 0.8 * np.exp(-W) * np.exp(2j * np.pi * h @ sites.x[0])
    )
    assert np.allclose(structure_factor(h, simple), expected)


def test_peak_integrates_to_f_squared(simple):
    """Summed over a fine grid in h, one peak carries |F(G)|^2."""

    profile = Profile(0.05, 0.01, 0.005)
    G = np.array([1.0, 2.0, 1.0])
    step = 0.01
    g = np.arange(-0.2, 0.2 + step / 2, step)
    grid = G + np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    I = bragg_design(grid, simple, profile)
    F2 = np.abs(structure_factor(G[None], simple)[0]) ** 2
    assert np.isclose(I.sum() * step**3, F2, rtol=1e-3)


def test_bin_average_conserves_intensity(simple):
    """Bin-averaged values on a bin-tiled grid integrate to |F|^2 too,
    for skewed bins of a projected volume."""

    profile = Profile(0.06, 0.0, 0.0)
    W = np.array([[1.0, -1, 0], [1, 1, 0], [0, 0, 1]])
    widths = np.array([0.04, 0.05, 0.06])
    G = np.array([2.0, 1.0, 1.0])
    u0 = np.linalg.solve(W, G)
    k = np.arange(-6, 7)
    u = (
        u0
        + np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
        * widths
    )
    I = bragg_design(
        u @ W.T, simple, profile, widths=widths, projection=W, samples=4
    )
    bin_volume = np.prod(widths) * abs(np.linalg.det(W))
    F2 = np.abs(structure_factor(G[None], simple)[0]) ** 2
    assert np.isclose(I.sum() * bin_volume, F2, rtol=2e-3)


def test_fit_profile_recovers_widths(simple):
    rng = np.random.default_rng(0)
    truth = Profile(0.04, 0.012, 0.006)
    G = np.array([[1, 0, 1], [2, -1, 1], [0, 2, 3], [3, 1, -2]], float)
    h = (G[:, None, :] + rng.normal(0, 0.06, (4, 400, 3))).reshape(-1, 3)
    y0 = 2.5 * bragg_design(h, simple, truth) + 0.3
    sigma = 0.01 * y0.max() * np.ones_like(y0)
    y = y0 + sigma * rng.normal(size=len(y0))
    profile, scale, chi2 = fit_profile(
        y,
        sigma,
        h,
        simple,
        Profile(0.06, 0.005, 0.01),
        background=np.ones((len(h), 1)),
    )
    assert np.isclose(scale, 2.5, rtol=0.02)
    assert np.isclose(profile.sigma0, 0.04, rtol=0.05)
    assert np.isclose(profile.radial, 0.012, rtol=0.15)
    assert 0.8 < chi2 < 1.2
