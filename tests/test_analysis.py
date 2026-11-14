import numpy as np
import pytest

from rmc_discord.analysis import correlations
from rmc_discord.analysis.correlations import _site_maps
from rmc_discord.model.supercell import Supercell


def toy_supercell(crystal, N=(3, 3, 2), rng=0):
    sc = Supercell(
        crystal, N, ["Fe1", "Co1"],
        ions={"Fe1": "Fe3+", "Mn1": "Mn3+", "Co1": "Co2+"},
        moments={"Fe3+": 5.0, "Mn3+": 4.0, "Co2+": 3.0},
        rng=rng,
    )
    sc.randomize_displacements()
    return sc


def brute_force(sc, quantity):
    """Pair sums over every atom pair, keyed by (site, site, dR mod N)."""
    vacant = np.array([k.vacancy for k in sc.kinds])
    magnetic = np.array([k.magnetic for k in sc.kinds])
    if quantity == "occupancy":
        mixed = {j: next(k for k in ks if not vacant[k]) for j, ks in enumerate(sc.site_kinds) if len(ks) > 1}
        on = np.array([j in mixed for j in sc.site])
        a = np.zeros((len(sc), 1))
        for i in np.flatnonzero(on):
            k = mixed[sc.site[i]]
            a[i, 0] = (sc.kind[i] == k) - sc.kinds[k].fraction
        m = on.astype(float)
    elif quantity == "spin":
        m = magnetic[sc.kind].astype(float)
        a = sc.S * m[:, None]
    else:
        m = (~vacant[sc.kind]).astype(float)
        a = sc.u * m[:, None]
    N = np.array(sc.N)
    num, den = {}, {}
    for i in range(len(sc)):
        if m[i] == 0:
            continue
        for k in range(len(sc)):
            if m[k] == 0:
                continue
            key = (sc.site[i], sc.site[k], *np.mod(sc.cell[k] - sc.cell[i], N))
            num[key] = num.get(key, 0.0) + a[i] @ a[k]
            den[key] = den.get(key, 0.0) + m[i] * m[k]
    return {key: num[key] / den[key] for key in num}, den


@pytest.mark.parametrize("quantity", ["occupancy", "spin", "displacement"])
def test_fft_matches_brute_force(toy_crystal, quantity):
    sc = toy_supercell(toy_crystal)
    corr = correlations(sc, quantity, symmetrize=False)
    expected, den = brute_force(sc, quantity)
    N = np.array(sc.N)
    found = {
        (j, k, *np.mod(d, N)): (v, n)
        for (j, k), d, v, n in zip(corr.sites, corr.dR, corr.value, corr.count)
    }
    assert set(found) == set(expected)
    for key, value in expected.items():
        assert found[key][0] == pytest.approx(value, abs=1e-12)
        assert found[key][1] == pytest.approx(den[key])
    # P 1: symmetrizing changes nothing
    sym = correlations(sc, quantity, symmetrize=True)
    assert np.allclose(sym.value, corr.value)


def transformed(sc, op):
    """Configuration moved by space-group operator ``op``."""
    channels = np.arange(sc.n_sites)
    W, sigma, t, Rc = _site_maps(sc, channels)[op]
    N = np.array(sc.N)
    kind = np.empty_like(sc.kind)
    u = np.empty_like(sc.u)
    for i in range(len(sc)):
        j = sc.site[i]
        cell = np.mod(W @ sc.cell[i] + t[j], N)
        target = np.ravel_multi_index(cell, N) * sc.n_sites + sigma[j]
        pos = list(sc.site_kinds[j]).index(sc.kind[i])
        kind[target] = sc.site_kinds[sigma[j]][pos]
        u[target] = Rc @ sc.u[i]
    return {"kind": kind, "u": u}


@pytest.mark.parametrize("quantity", ["occupancy", "displacement"])
def test_symmetrized_correlations_are_invariant(carbonate_crystal, quantity):
    sc = Supercell(carbonate_crystal, (3, 3, 2), ["C2"], rng=2)
    sc.randomize_displacements()
    assert len(carbonate_crystal.operators) == 6
    identity = [i for i, (W, w) in enumerate(carbonate_crystal.operators)
                if np.array_equal(W, np.eye(3)) and not np.any(w)]
    for op in [i for i in range(6) if i not in identity]:
        moved = transformed(sc, op)
        a = correlations(sc, quantity)
        b = correlations(sc, quantity, states=[moved])
        assert np.allclose(a.value, b.value, atol=1e-12)
        if quantity == "displacement":
            assert np.allclose(a.tensor, b.tensor, atol=1e-12)
        raw_a = correlations(sc, quantity, symmetrize=False)
        raw_b = correlations(sc, quantity, states=[moved], symmetrize=False)
        assert not np.allclose(raw_a.value, raw_b.value)


def test_pair_reversal_identity(carbonate_crystal):
    sc = Supercell(carbonate_crystal, (3, 3, 2), ["C2"], rng=2)
    corr = correlations(sc, "occupancy", symmetrize=False)
    N = np.array(sc.N)
    table = {(j, k, *np.mod(d, N)): v for (j, k), d, v in zip(corr.sites, corr.dR, corr.value)}
    for (j, k, *d), v in table.items():
        assert table[(k, j, *np.mod(-np.array(d), N))] == pytest.approx(v)


def test_antiferromagnetic_pattern(toy_crystal):
    sc = toy_supercell(toy_crystal, N=(4, 3, 2))
    sign = np.where(sc.cell[:, 0] % 2 == 0, 1.0, -1.0)
    sc.S[:] = np.array([0.0, 0.0, 1.0]) * sign[:, None]
    corr = correlations(sc, "spin")
    fe = np.flatnonzero(np.array([s.labels[0] == "Fe1" for s in sc.sites]))[0]
    pick = (corr.sites[:, 0] == fe) & (corr.sites[:, 1] == fe)
    for d, expected in (((1, 0, 0), -1.0), ((2, 0, 0), 1.0), ((0, 1, 0), 1.0), ((1, 1, 1), -1.0)):
        hit = pick & np.all(corr.dR == d, axis=1)
        assert corr.value[hit] == pytest.approx([expected])
    shells = corr.shells(r_max=4.5)
    i = np.flatnonzero((shells["label"] == "Fe1/Mn1-Fe1/Mn1") & np.isclose(shells["distance"], 4.0))
    assert shells["value"][i] == pytest.approx([0.0], abs=1e-12)  # (+-1,0,0): -1, (0,+-1,0): +1


def test_normalized_and_errors(toy_crystal):
    sc = toy_supercell(toy_crystal)
    states = []
    for seed in range(3):
        other = toy_supercell(toy_crystal, rng=seed + 10)
        states.append({"kind": other.kind, "u": other.u, "S": other.S})
    occ = correlations(sc, "occupancy", states=states, normalize=True)
    self_term = occ.select((occ.distance < 1e-8) & (occ.sites[:, 0] == occ.sites[:, 1]))
    c = sc.kinds[sc.site_kinds[self_term.sites[0, 0]][0]].fraction
    assert np.all(np.isfinite(occ.value))
    assert np.all(occ.error >= 0) and np.any(occ.error > 0)
    disp = correlations(sc, "displacement", normalize=True)
    self_disp = disp.select((disp.distance < 1e-8) & (disp.sites[:, 0] == disp.sites[:, 1]))
    assert self_disp.value == pytest.approx(np.ones(len(self_disp)))
    assert disp.tensor.shape == (len(disp), 3, 3)
    assert 0 < c < 1
