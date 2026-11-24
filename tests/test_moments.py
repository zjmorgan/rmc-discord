import numpy as np
import pytest

from rmc_discord.crystal.moments import magnetic_ions, parse_ion, shell, spin_states


@pytest.mark.parametrize(
    "ion, moment",
    [("Fe3+", 5.92), ("Mn3+", 4.90), ("Mn2+", 5.92), ("Co2+", 3.87), ("Ni2+", 2.83), ("Cu2+", 1.73),
     ("Cr3+", 3.87)],
)
def test_spin_only_high_spin(ion, moment):
    high = [s for s in spin_states(ion) if s.name.startswith("high spin")]
    assert np.isclose(high[0].moment, moment, atol=0.01)


@pytest.mark.parametrize(
    "ion, J, g, moment",
    [("Ce3+", 2.5, 6 / 7, 2.54), ("Gd3+", 3.5, 2.0, 7.94), ("Tb3+", 6.0, 1.5, 9.72), ("Dy3+", 7.5, 4 / 3, 10.65),
     ("Er3+", 7.5, 6 / 5, 9.58), ("Yb3+", 3.5, 8 / 7, 4.54)],
)
def test_rare_earth_hund(ion, J, g, moment):
    (state,) = spin_states(ion)
    assert state.J == J and np.isclose(state.g, g) and np.isclose(state.moment, moment, atol=0.01)


def test_states_and_ions():
    names = [s.name for s in spin_states("Fe2+")]
    assert names == ["high spin, S = 2"]  # low-spin d6 carries no moment
    co = spin_states("Co3+")
    assert [s.S for s in co] == [2.0]
    fe = spin_states("Fe3+")  # low spin first (the default), then high spin
    assert [s.S for s in fe] == [0.5, 2.5] and np.isclose(fe[0].moment, np.sqrt(3))
    assert [s.S for s in spin_states("Cr3+")] == [1.5]  # d3: low and high spin coincide
    ir = spin_states("Ir4+")
    assert ir[0].name.startswith("low spin") and np.isclose(ir[0].moment, np.sqrt(3))
    assert spin_states("Eu3+") == [] and spin_states("Zn2+") == []
    assert shell("Fe3+") == ("3d", 5) and shell("U4+") == ("5f", 2)
    assert parse_ion("O2-") == ("O", -2) and parse_ion("Fe") == ("Fe", 0)
    assert "Fe3+" in magnetic_ions("Fe") and magnetic_ions("O") == []
