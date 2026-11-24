import numpy as np
import pytest

from rmc_discord.pairs.io import load_mdhisto, reduce_volume

h5py = pytest.importorskip("h5py")


def write_mdhisto(path, signal, errors_squared, edges):
    """Minimal synthetic file laid out like Mantid SaveMD output: the
    stored array is ordered D2, D1, D0."""

    with h5py.File(path, "w") as f:
        g = f.create_group("MDHistoWorkspace/data")
        s = g.create_dataset("signal", data=np.transpose(signal, (2, 1, 0)))
        s.attrs["axes"] = "D2:D1:D0"
        g.create_dataset(
            "errors_squared", data=np.transpose(errors_squared, (2, 1, 0))
        )
        for k, e in enumerate(edges):
            g.create_dataset(f"D{k}", data=e)


def test_axis_order_and_reduction(tmp_path):
    rng = np.random.default_rng(0)
    shape = (4, 3, 2)
    edges = [np.linspace(-1, 1, n + 1) for n in shape]
    arrays = {}
    for name in ("data", "norm", "bkg_data", "bkg_norm"):
        signal = rng.random(shape) + 0.5
        arrays[name] = signal
        write_mdhisto(tmp_path / f"{name}.nxs", signal, signal / 10, edges)

    signal, errors_squared, e, W = load_mdhisto(tmp_path / "data.nxs")
    assert signal.shape == shape
    assert np.allclose(signal, arrays["data"])
    assert np.allclose(W, np.eye(3))
    assert all(np.allclose(a, b) for a, b in zip(e, edges))

    paths = [tmp_path / f"{n}.nxs" for n in arrays]
    volume = reduce_volume(*paths, scale=0.5)
    a = arrays
    expected = a["data"] / a["norm"] - 0.5 * a["bkg_data"] / a["bkg_norm"]
    assert np.allclose(volume.signal, expected)
    assert np.allclose(volume.widths, [0.5, 2 / 3, 1.0])
    index = volume.index(np.array([[0.1, -0.9, 0.5]]))
    assert np.allclose(volume.hkl(index), [[0.25, -2 / 3, 0.5]])


def test_reduction_without_background(tmp_path):
    rng = np.random.default_rng(1)
    shape = (3, 2, 2)
    edges = [np.linspace(-1, 1, n + 1) for n in shape]
    d, n = rng.random(shape) + 0.5, rng.random(shape) + 0.5
    n[0, 0, 0] = 0
    write_mdhisto(tmp_path / "d.nxs", d, d / 10, edges)
    write_mdhisto(tmp_path / "n.nxs", n, n / 10, edges)
    volume = reduce_volume(tmp_path / "d.nxs", tmp_path / "n.nxs")
    valid = n > 0
    assert np.all(np.isnan(volume.signal[~valid]))
    dv, nv = d[valid], n[valid]
    assert np.allclose(volume.signal[valid], dv / nv)
    assert np.allclose(volume.sigma[valid] ** 2, dv / 10 / nv**2)
    with pytest.raises(ValueError):
        reduce_volume(
            tmp_path / "d.nxs", tmp_path / "n.nxs", tmp_path / "d.nxs"
        )


def test_lattice_and_with_cell(tmp_path, tetragonal):
    from rmc_discord.pairs.io import load_lattice

    path = tmp_path / "lattice.nxs"
    with h5py.File(path, "w") as f:
        base = "MDHistoWorkspace/experiment0/sample/oriented_lattice/"
        for k, v in zip(
            ("a", "b", "c", "alpha", "beta", "gamma"),
            (5.1, 5.1, 4.2, 90, 90, 90),
        ):
            f.create_dataset(base + "unit_cell_" + k, data=[v])
    cell = load_lattice(path)
    assert cell == (5.1, 5.1, 4.2, 90, 90, 90)
    moved = tetragonal.with_cell(cell)
    assert np.allclose(np.linalg.norm(moved.A, axis=0), [5.1, 5.1, 4.2])
    a, b = tetragonal.sites(), moved.sites()
    assert np.allclose(a.x, b.x) and np.allclose(a.U, b.U)
