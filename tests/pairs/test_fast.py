import numpy as np
import pytest

from rmc_discord.data.volume import Volume
from rmc_discord.pairs.fast import forward
from rmc_discord.pairs.workflow import ModelSpec, PairModel

from test_workflow import synthetic

pytest.importorskip("finufft")

SPECS = {
    "structural": dict(r_max=6.0, cluster_distance=1.0, size_effect=True, displacement=True),
    "isotropic": dict(active=[], r_max=7.0, magnetic="isotropic", magnetic_sites=["M1"], ions={"M1": "Ni2+"},
                      moments={"Ni2+": 2.0}),
    "tensor": dict(active=[], r_max=6.0, magnetic="tensor", magnetic_sites=["M1"], ions={"M1": "Ni2+"},
                   moments={"Ni2+": 2.0}),
}


def fitted(structure, name, seed=0):
    model = PairModel(structure, ModelSpec(**SPECS[name]))
    obs, _ = synthetic(model, np.random.default_rng(seed), m=800)
    return model, model.fit(obs, background="polynomial", degree=1), obs


@pytest.mark.parametrize("name", list(SPECS))
def test_points_match_design(tetragonal, name):
    model, fit, obs = fitted(tetragonal, name)
    h = np.random.default_rng(5).uniform(-4, 4, (300, 3))
    ref = model.predict(fit, h, obs.widths, obs.projection)
    fast = forward(model, fit, h=h, widths=obs.widths, projection=obs.projection, eps=1e-10)
    assert np.allclose(fast, ref, rtol=0, atol=1e-8 * np.abs(ref).max())


@pytest.mark.parametrize("name", ["structural", "isotropic"])
def test_volume_matches_design(tetragonal, name):
    model, fit, _ = fitted(tetragonal, name, seed=1)
    model.operators = np.array([np.eye(3), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]])  # extra mirror average
    W = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # skewed volume axes
    edges = [np.linspace(-2.0, 2.0, 22), np.linspace(-3.0, 3.1, 30), np.linspace(-1.5, 2.5, 17)]
    shape = tuple(len(e) - 1 for e in edges)
    vol = Volume(np.zeros(shape), np.ones(shape), edges, W)
    fast = forward(model, fit, volume=vol, eps=1e-10)
    index = np.argwhere(np.ones(shape, bool))
    ref = model.predict(fit, vol.hkl(index), vol.widths, vol.W).reshape(shape)
    assert np.allclose(fast, ref, rtol=0, atol=1e-8 * np.abs(ref).max())
    only = forward(model, fit, volume=vol, channels=[model.blocks[0].name], background=False)
    assert not np.allclose(only, fast)
