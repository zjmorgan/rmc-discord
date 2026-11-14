import numpy as np
import pytest

from rmc_discord.data import sampling, volume
from rmc_discord.data.volume import Volume


def synthetic_volume(W, step=0.05, extent=1.5, seed=0):
    n = int(round(2 * extent / step))
    edges = [np.linspace(-extent, extent, n + 1) for _ in range(3)]
    rng = np.random.default_rng(seed)
    signal = rng.random((n, n, n)) + 1.0
    variance = rng.random((n, n, n)) * 0.1 + 0.01
    signal[0, 0, 0] = np.nan
    return Volume(signal=signal, variance=variance, edges=edges, W=np.asarray(W, float))


@pytest.mark.parametrize("W", [np.eye(3), [[1, -1, 0], [1, 1, 0], [0, 0, 1]]])
def test_commensurate_mean_combines_all_bins(toy_crystal, W):
    vol = synthetic_volume(W)
    N = (4, 4, 3)
    sample = sampling.commensurate(vol, toy_crystal, N, aggregate="mean")

    assert np.allclose(sample.hkl * N, sample.m)
    assert not np.any(np.all(sample.m % N == 0, axis=1))

    valid = np.isfinite(vol.signal)
    hkl = vol.hkl(np.argwhere(valid))
    m = np.round(hkl * N).astype(int)
    integer = np.all(m % N == 0, axis=1)
    assert sample.bins.sum() == np.count_nonzero(~integer)

    k = len(sample) // 2
    members = np.all(m == sample.m[k], axis=1)
    w = 1 / vol.variance[valid][members]
    y = vol.signal[valid][members]
    assert np.isclose(sample.y[k], np.sum(w * y) / np.sum(w))
    assert np.isclose(sample.variance[k], 1 / np.sum(w))


def test_commensurate_nearest_uses_containing_bin(toy_crystal):
    vol = synthetic_volume(np.eye(3), step=0.1)
    N = (5, 5, 5)
    sample = sampling.commensurate(vol, toy_crystal, N, aggregate="nearest")
    assert np.all(sample.bins == 1)
    index, inside = vol.index(sample.hkl)
    assert np.all(inside)
    assert np.allclose(sample.y, vol.signal[tuple(index.T)])


def test_region_of_interest(toy_crystal):
    vol = synthetic_volume(np.eye(3))
    roi = [[0, 1], [0, 1], [0, 0.5]]
    sample = sampling.commensurate(vol, toy_crystal, (4, 4, 4), hkl_range=roi)
    assert np.all(sample.hkl >= -1e-12)
    assert np.all(sample.hkl <= np.array([1, 1, 0.5]) + 1e-12)


def test_bragg_mask_uses_reflection_conditions():
    from conftest import LEGACY_DATA
    from rmc_discord.crystal.structure import Crystal

    crystal = Crystal(
        (4.0, 4.0, 4.0, 90, 90, 90), "I m -3 m",
        [__import__("rmc_discord.crystal.cif", fromlist=["AsymmetricSite"])
         .AsymmetricSite("Fe1", "Fe", np.zeros(3), 1.0)],
    )
    hkl = np.array([[1.1, 0.0, 0.0], [1.1, 1.0, 0.0], [1.5, 1.0, 0.0]])
    sample = sampling.QSample(
        m=(hkl * 10).astype(int), N=(10, 10, 10), hkl=hkl, Q=crystal.Q(hkl),
        y=np.ones(3), variance=np.ones(3), bins=np.ones(3, int),
    )
    near = sampling.bragg_mask(sample, crystal, radius=0.15)
    assert near.tolist() == [False, True, False]
    near_all = sampling.bragg_mask(sample, crystal, radius=0.15, allowed_only=False)
    assert near_all.tolist() == [True, True, False]


def test_save_load_round_trip(tmp_path):
    vol = synthetic_volume([[1, -1, 0], [1, 1, 0], [0, 0, 1]], step=0.25)
    filename = tmp_path / "volume.nxs"
    volume.save(vol, filename)
    back = volume.load(filename)
    ok = np.isfinite(vol.signal)
    assert np.allclose(back.signal[ok], vol.signal[ok])
    assert np.allclose(back.variance[ok], vol.variance[ok])
    assert np.allclose(back.W, vol.W)
    for a in range(3):
        assert np.allclose(back.edges[a], vol.edges[a], atol=1e-5)


def test_load_bixbyite(private_data):
    files = sorted((private_data / "bixbyite_CORELLI").glob("bixbyite_006K_*.nxs"))
    vol = volume.load(files[0])
    assert vol.shape == (201, 201, 201)
    assert np.allclose(vol.W, np.eye(3))
    assert np.allclose(vol.widths, 0.1, atol=1e-6)
    assert np.allclose(vol.hkl([[100, 100, 100]]), 0, atol=1e-6)
    assert 0.9 < np.isfinite(vol.signal).mean() < 1.0


def test_commensurate_counts_and_norm(toy_crystal):
    rng = np.random.default_rng(3)
    n = 30
    edges = [np.linspace(-1.5, 1.5, n + 1) for _ in range(3)]
    norm = rng.random((n, n, n)) * 10 + 1
    counts = rng.poisson(2.0 * norm).astype(float)
    data = Volume(signal=counts, variance=counts.copy(), edges=edges, W=np.eye(3))
    normalization = Volume(signal=norm, variance=np.zeros_like(norm), edges=edges, W=np.eye(3))
    N = (4, 4, 3)
    sample = sampling.commensurate(data, toy_crystal, N, norm=normalization)

    hkl = data.hkl(np.argwhere(np.ones(data.shape, bool)))
    m = np.round(hkl * N).astype(int)
    k = len(sample) // 3
    members = np.all(m == sample.m[k], axis=1)
    d, w = counts.ravel()[members], norm.ravel()[members]
    assert np.isclose(sample.y[k], d.sum() / w.sum())
    assert np.isclose(sample.variance[k], max(d.sum(), 1) / w.sum() ** 2)
    assert np.isclose(np.average(sample.y, weights=sample.bins), 2.0, rtol=0.02)

    ratio = data / normalization
    assert np.allclose(ratio.signal, counts / norm)
    assert np.allclose(ratio.variance, counts / norm**2)
