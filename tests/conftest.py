import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmc_discord.crystal.cif import AsymmetricSite  # noqa: E402
from rmc_discord.crystal.structure import Crystal  # noqa: E402

LEGACY_DATA = ROOT / "disorder" / "tests" / "data"

# pair-correlation tests (ported from discordant-parsimony)
from rmc_discord.pairs import Structure as PairStructure  # noqa: E402
from rmc_discord.pairs.io import data_root as pairs_data_root  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "pairs" / "fixtures"
TB5RU6SN18 = pairs_data_root() / "Tb5Ru6Sn18_293K"
requires_tb5ru6sn18 = pytest.mark.skipif(
    not (TB5RU6SN18 / "Tb5Ru6Sn18.cif").exists(),
    reason="private Tb5Ru6Sn18 data not available (set DIFFUSE_DATA_ROOT)",
)


@pytest.fixture
def tetragonal():
    return PairStructure.from_cif(FIXTURES / "split_tetragonal.cif")


@pytest.fixture
def tb5ru6sn18():
    return PairStructure.from_cif(TB5RU6SN18 / "Tb5Ru6Sn18.cif")


def data_root():
    root = os.environ.get("DIFFUSE_DATA_ROOT")
    if root:
        return Path(root)
    sibling = ROOT.parent / "discordant-parsimony" / "local_data"
    return sibling if sibling.exists() else None


@pytest.fixture(scope="session")
def private_data():
    root = data_root()
    if root is None:
        pytest.skip("private data not available (set DIFFUSE_DATA_ROOT)")
    return root


@pytest.fixture(scope="session")
def toy_crystal():
    """Tetragonal cell with a mixed, a magnetic, and a displaced site."""

    U = np.diag([0.010, 0.012, 0.008])
    sites = [
        AsymmetricSite("Fe1", "Fe", np.array([0.0, 0.0, 0.0]), 0.6, U=None, U_iso=0.005),
        AsymmetricSite("Mn1", "Mn", np.array([0.0, 0.0, 0.0]), 0.3, U=None, U_iso=0.005),
        AsymmetricSite("Co1", "Co", np.array([0.5, 0.5, 0.25]), 1.0, U=U, U_iso=0.01),
        AsymmetricSite("O1", "O", np.array([0.5, 0.0, 0.1]), 1.0, U=None, U_iso=0.008),
    ]
    return Crystal((4.0, 4.0, 5.5, 90, 90, 90), "P 1", sites)


@pytest.fixture(scope="session")
def carbonate_crystal():
    """P-6 cell with two partially occupied CO3 groups per cell (2h + 6l)."""

    sites = [
        AsymmetricSite("Ba1", "Ba", np.array([0.335, 0.012, 0.0]), 1.0, U=None, U_iso=0.004),
        AsymmetricSite("C2", "C", np.array([1 / 3, 2 / 3, 0.187]), 0.8, U=None, U_iso=0.008),
        AsymmetricSite("O8", "O", np.array([0.203, 0.529, 0.187]), 0.8, U=None, U_iso=0.008),
    ]
    return Crystal((9.683, 9.683, 9.518, 90, 90, 120), "P -6", sites)
