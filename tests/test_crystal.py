import numpy as np
import pytest

from conftest import LEGACY_DATA
from rmc_discord.crystal import tables
from rmc_discord.crystal.structure import Crystal


def test_phase_convention(toy_crystal):
    rng = np.random.default_rng(0)
    h = rng.normal(size=(5, 3))
    x = rng.normal(size=(5, 3))
    Qr = np.sum(toy_crystal.Q(h) * toy_crystal.cartesian(x), axis=1)
    assert np.allclose(Qr, 2 * np.pi * np.sum(h * x, axis=1))


def test_mixed_site_grouping(toy_crystal):
    assert toy_crystal.n_atoms == 4
    assert len(toy_crystal.sites) == 3
    mixed = toy_crystal.sites[toy_crystal.site_indices(["Fe1"])[0]]
    assert mixed.labels == ["Fe1", "Mn1"]
    assert np.isclose(mixed.vacancy, 0.1)


@pytest.mark.parametrize(
    "name, laue, n_atoms, n_sites",
    [("Ba3Co2O6(CO3)0.6.cif", "6/m", 56, 56)],
)
def test_cif_expansion(name, laue, n_atoms, n_sites):
    crystal = Crystal.from_cif(LEGACY_DATA / name)
    assert crystal.laue == laue
    assert crystal.n_atoms == n_atoms
    assert len(crystal.sites) == n_sites


def test_bixbyite(private_data):
    crystal = Crystal.from_cif(private_data / "bixbyite_CORELLI" / "bixbyite.cif")
    assert crystal.space_group.getNumber() == 206
    assert crystal.n_atoms == 112
    assert len(crystal.sites) == 80

    labels = [tuple(s.labels) for s in crystal.sites]
    assert labels.count(("Fe1", "Mn1")) == 8
    assert labels.count(("Mn2", "Fe2")) == 24

    for orbit in {s.orbit for s in crystal.sites}:
        U = [s.U for s in crystal.sites if s.orbit == orbit]
        eig = [np.linalg.eigvalsh(u) for u in U]
        assert np.allclose(eig, eig[0], atol=1e-10)


def test_hexagonal_adp_is_isotropic_in_cartesian():
    crystal = Crystal.from_cif(LEGACY_DATA / "Ba3Co2O6(CO3)0.6.cif")
    for site in crystal.sites:
        assert np.allclose(site.U, site.U[0, 0] * np.eye(3), atol=1e-12)


def test_tables():
    assert np.isclose(tables.scattering_length("Fe").real, 9.45)
    assert np.isclose(tables.scattering_length("Fe3+").real, 9.45)
    f = tables.magnetic_form_factor("Fe3+", np.array([0.0, 4.0]))
    assert np.isclose(f[0], 1.0, atol=0.01) and 0 < f[1] < 1
