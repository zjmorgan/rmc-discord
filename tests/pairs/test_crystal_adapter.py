"""Structure.from_crystal agrees with the CIF reader up to the Cartesian frame."""

from pathlib import Path

import numpy as np
import pytest

from rmc_discord.crystal.structure import Crystal
from rmc_discord.pairs import Structure
from rmc_discord.pairs.io import data_root
from rmc_discord.pairs.design import design_matrix
from rmc_discord.pairs.symmetry import pair_orbits

LEGACY = Path(__file__).resolve().parents[2] / "disorder" / "tests" / "data"
CIFS = [LEGACY / "Ba3Co2O6(CO3)0.6.cif", LEGACY / "Tb2Ir3Ga9.cif", LEGACY / "natrolite.cif",
        data_root() / "bixbyite_CORELLI" / "bixbyite.cif", data_root() / "Tb5Ru6Sn18_293K" / "Tb5Ru6Sn18.cif"]


@pytest.mark.parametrize("path", CIFS, ids=lambda p: p.name)
def test_from_crystal_matches_cif_reader(path):
    if not path.exists():
        pytest.skip("private data not available (set DIFFUSE_DATA_ROOT)")
    crystal = Crystal.from_cif(str(path))
    a = Structure.from_crystal(crystal)
    b = Structure.from_cif(path)
    R = a.rotation
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(a.A, crystal.A) and np.allclose(a.A, R @ b.A)
    assert np.allclose(a.B, crystal.B)

    sa, sb = a.sites(), b.sites()
    key = lambda s: sorted(zip(s.label, np.round(s.x, 5).tolist(), np.round(s.occupancy, 6)))
    assert key(sa) == key(sb)
    order_a = np.lexsort(np.vstack([np.round(sa.x, 5).T, sa.label]))
    order_b = np.lexsort(np.vstack([np.round(sb.x, 5).T, sb.label]))
    assert np.array_equal(sa.species[order_a], sb.species[order_b])
    Ua, Ub = sa.U[order_a], R @ sb.U[order_b] @ R.T
    assert np.allclose(Ua, Ub, atol=1e-8)

    h = np.random.default_rng(0).uniform(-3, 3, (200, 3))
    assert np.allclose(np.linalg.norm(a.Q(h), axis=1), np.linalg.norm(b.Q(h), axis=1))
    assert np.allclose(a.Q(h), b.Q(h) @ R.T)

    labels = sa.partial() or [sa.label[0]]
    act_a, act_b = sa.select(labels), sb.select(labels)
    oa, ob = pair_orbits(a, act_a, 6.0), pair_orbits(b, act_b, 6.0)
    assert oa.n_orbits == ob.n_orbits
    Aa = design_matrix(h, oa, act_a, a)
    Ab = design_matrix(h, ob, act_b, b)
    assert np.allclose(np.sort(Aa.sum(0)), np.sort(Ab.sum(0)), rtol=1e-10, atol=1e-10)
    assert np.allclose(Aa.sum(1), Ab.sum(1), rtol=1e-10, atol=1e-10)


def test_frame_survives_cell_and_occupancy_changes():
    crystal = Crystal.from_cif(str(LEGACY / "Ba3Co2O6(CO3)0.6.cif"))
    s = Structure.from_crystal(crystal)
    label = crystal.asymmetric[0].label
    t = s.with_occupancies({label: 0.5}).with_cell(crystal.cell)
    assert np.allclose(t.A, s.A) and np.allclose(t.rotation, s.rotation)
