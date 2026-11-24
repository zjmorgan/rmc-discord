import numpy as np
import pytest

from rmc_discord.pairs import design
from rmc_discord.pairs.workflow import ModelSpec, PairModel

from test_fast import SPECS


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("thermal", [True, False])
def test_fused_matches_numpy(tetragonal, monkeypatch, name, thermal):
    model = PairModel(tetragonal, ModelSpec(**SPECS[name], thermal=thermal))
    model.operators = np.array([np.eye(3), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]])
    h = np.random.default_rng(3).uniform(-4, 4, (257, 3))
    W = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    widths = np.array([0.05, 0.03, 0.08])
    monkeypatch.setattr(design, "BACKEND", "numpy")
    ref = model.design(h, widths, W)
    monkeypatch.setattr(design, "BACKEND", "numba")
    fast = model.design(h, widths, W)
    assert ref.shape == fast.shape
    assert np.allclose(fast, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
