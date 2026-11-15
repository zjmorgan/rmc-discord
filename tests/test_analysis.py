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


def test_save_load_and_selection(toy_crystal, tmp_path):
    from rmc_discord.analysis import Correlations, read_shells, write_shells

    sc = toy_supercell(toy_crystal)
    for quantity in ("occupancy", "displacement"):
        corr = correlations(sc, quantity)
        corr.save(tmp_path / "c.h5")
        back = Correlations.load(tmp_path / "c.h5")
        assert back.quantity == quantity and back.labels == corr.labels
        for key in ("sites", "dR", "r", "value", "error", "count"):
            assert np.array_equal(getattr(back, key), getattr(corr, key))
        assert (back.tensor is None) == (corr.tensor is None)
    fe = corr.pair("Fe1")
    assert len(fe) > 0 and all(label == "Fe1/Mn1-Fe1/Mn1" for label in fe.labels)
    mixed = corr.pair("Fe1", "Co1")
    assert all(set(label.split("-")) == {"Fe1/Mn1", "Co1"} for label in mixed.labels)
    assert np.all(corr.within(5.0).distance <= 5.0)
    shells = corr.shells(r_max=6.0)
    write_shells(shells, tmp_path / "s.csv")
    back = read_shells(tmp_path / "s.csv")
    assert np.array_equal(back["label"], shells["label"])
    assert np.allclose(back["value"], shells["value"]) and np.array_equal(back["n"], shells["n"])


def commensurate_points(crystal, N):
    from rmc_discord.data.sampling import QSample

    N = np.array(N)
    m = np.stack(np.meshgrid(*[np.arange(-n, n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    m = m[~np.all(m % N == 0, axis=1)]
    hkl = m / N
    n = len(m)
    return QSample(m=m, N=tuple(N), hkl=hkl, Q=crystal.Q(hkl), y=np.ones(n),
                   variance=np.ones(n), bins=np.ones(n, dtype=int))


def test_occupancy_recalculation_is_exact(carbonate_crystal):
    from rmc_discord.analysis import recalculate
    from rmc_discord.engine.refinement import Refinement

    sc = Supercell(carbonate_crystal, (3, 3, 2), ["C2"], rng=2)
    sample = commensurate_points(carbonate_crystal, sc.N)
    expected = Refinement(sc, sample, fit="none").intensity()
    corr = correlations(sc, "occupancy", symmetrize=False)
    assert np.allclose(recalculate(sc, corr, sample.hkl), expected, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        recalculate(sc, correlations(sc, "occupancy", normalize=True), sample.hkl)
    # truncation keeps a real, finite intensity
    near = recalculate(sc, corr, sample.hkl, r_max=6.0)
    assert np.all(np.isfinite(near)) and not np.allclose(near, expected)


def test_spin_recalculation_is_exact():
    from rmc_discord.analysis import recalculate
    from rmc_discord.crystal.cif import AsymmetricSite
    from rmc_discord.crystal.structure import Crystal
    from rmc_discord.engine.refinement import Refinement

    crystal = Crystal((4.0, 4.5, 5.0, 90, 90, 90), "P 1",
                      [AsymmetricSite("Mn1", "Mn", np.zeros(3), 1.0, U_iso=0.004),
                       AsymmetricSite("Mn2", "Mn", np.array([0.5, 0.3, 0.2]), 1.0, U_iso=0.006)])
    sc = Supercell(crystal, (3, 3, 2), ["Mn1", "Mn2"], ions={"Mn1": "Mn2+", "Mn2": "Mn3+"},
                   moments={"Mn2+": 5.0, "Mn3+": 4.0}, rng=4)
    sample = commensurate_points(crystal, sc.N)
    expected = Refinement(sc, sample, nuclear=False, magnetic=True, fit="none").intensity()
    corr = correlations(sc, "spin")
    assert corr.tensor.shape == (len(corr), 3, 3)
    assert np.allclose(np.trace(corr.tensor, axis1=1, axis2=2), corr.value)
    assert np.allclose(recalculate(sc, corr, sample.hkl), expected, rtol=1e-10, atol=1e-12)


def test_symmetrized_recalculation_averages_equivalent_q(carbonate_crystal):
    from rmc_discord.analysis import recalculate

    from rmc_discord.analysis import max_range

    sc = Supercell(carbonate_crystal, (4, 4, 4), ["C2"], rng=2)
    raw = correlations(sc, "occupancy", symmetrize=False)
    sym = correlations(sc, "occupancy")
    ops = carbonate_crystal.operators
    # commensurate points: exact for the full table
    hkl = commensurate_points(carbonate_crystal, sc.N).hkl[::7]
    avg = np.mean([recalculate(sc, raw, hkl @ W) for W, _ in ops], axis=0)
    assert np.allclose(recalculate(sc, sym, hkl), avg, rtol=1e-10, atol=1e-12)
    # any Q: exact once truncated inside the supercell
    r = max_range(sc)
    a, c = carbonate_crystal.cell[0], carbonate_crystal.cell[2]
    assert r == pytest.approx(min(4 * a * np.sqrt(3) / 2, 4 * c) / 2)  # hexagonal d_100 = a sqrt(3)/2
    hkl = np.random.default_rng(0).uniform(-3, 3, size=(40, 3))
    avg = np.mean([recalculate(sc, raw, hkl @ W, r_max=0.99 * r) for W, _ in ops], axis=0)
    assert np.allclose(recalculate(sc, sym, hkl, r_max=0.99 * r), avg, rtol=1e-10, atol=1e-12)


def test_displacement_recalculation_without_displacements_is_exact(carbonate_crystal):
    from rmc_discord.analysis import recalculate
    from rmc_discord.engine.refinement import Refinement

    sc = Supercell(carbonate_crystal, (3, 3, 2), ["C2"], rng=2)
    sc.u[:] = 0.0
    sample = commensurate_points(carbonate_crystal, sc.N)
    expected = Refinement(sc, sample, displacive=range(sc.n_sites), fit="none").intensity()
    corr = correlations(sc, "displacement", symmetrize=False)
    assert np.allclose(recalculate(sc, corr, sample.hkl), expected, rtol=1e-10, atol=1e-12)


def test_displacement_recalculation_matches_gaussian_ensemble(carbonate_crystal):
    """Correlated Gaussian displacements around vacancies: recalculation = ensemble mean."""
    from rmc_discord.analysis import recalculate
    from rmc_discord.engine.refinement import Refinement

    rng = np.random.default_rng(7)
    sc = Supercell(carbonate_crystal, (3, 3, 2), ["C2"], rng=2)
    sample = commensurate_points(carbonate_crystal, sc.N)
    sample = sample.select(sample.Q_norm < 5.0)
    N = np.array(sc.N)
    # neighbor along a with the same site
    nbr = np.array([np.ravel_multi_index(np.mod(c + [1, 0, 0], N), N) * sc.n_sites + s
                    for c, s in zip(sc.cell, sc.site)])
    states, intensities = [], []
    for _ in range(400):
        g = rng.normal(scale=0.2, size=(len(sc), 3))
        sc.u[:] = (g + 0.8 * g[nbr]) / np.sqrt(1 + 0.8**2)
        states.append({"kind": sc.kind.copy(), "u": sc.u.copy()})
        intensities.append(Refinement(sc, sample, displacive=range(sc.n_sites), fit="none").intensity())
    mean = np.mean(intensities, axis=0)
    error = np.std(intensities, axis=0) / np.sqrt(len(intensities))
    corr = correlations(sc, "displacement", states=states, symmetrize=False)
    I = recalculate(sc, corr, sample.hkl)
    z = (I - mean) / error
    assert np.abs(z).mean() < 1.5 and np.abs(np.mean(z)) < 0.3
    # control: dropping the inter-atomic displacement correlations is clearly wrong
    from dataclasses import replace

    tensor = corr.tensor.copy()
    tensor[~((corr.distance == 0) & (corr.sites[:, 0] == corr.sites[:, 1]))] = 0.0
    z0 = (recalculate(sc, replace(corr, tensor=tensor), sample.hkl) - mean) / error
    assert np.abs(z0).mean() > 3 * np.abs(z).mean()
