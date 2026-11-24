import numpy as np
import pytest

from rmc_discord.data.sampling import QSample
from rmc_discord.engine import kernels
from rmc_discord.engine.refinement import Refinement
from rmc_discord.model import scatterers
from rmc_discord.model.supercell import Supercell

N = (3, 3, 2)


def all_points(crystal, N, include_integer=False):
    """Every commensurate point in one reciprocal cell."""
    N = np.array(N)
    m = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), -1).reshape(-1, 3)
    m = m + N * np.array([1, -1, 2])
    if not include_integer:
        m = m[~np.all(m % N == 0, axis=1)]
    hkl = m / N
    n = len(m)
    return QSample(
        m=m, N=tuple(N), hkl=hkl, Q=crystal.Q(hkl),
        y=np.ones(n), variance=np.ones(n), bins=np.ones(n, dtype=int),
    )


def supercell(crystal, rng=0):
    sc = Supercell(
        crystal, N, ["Fe1", "Co1"],
        ions={"Fe1": "Fe3+", "Mn1": "Mn3+", "Co1": "Co2+"},
        moments={"Fe3+": 5.0, "Mn3+": 4.0, "Co2+": 3.0},
        rng=rng,
    )
    sc.randomize_displacements()
    return sc


def brute_force(sc, sample, P, M):
    x = sc.positions()
    phase = np.exp(1j * sample.Q @ x.T)
    F = np.sum(P[sc.kind].T * phase, axis=1)
    FM = np.einsum("qi,iq,ia->qa", phase, M[sc.kind], sc.S)
    Qhat = sample.Q / np.linalg.norm(sample.Q, axis=1, keepdims=True)
    FM_perp = FM - Qhat * np.sum(Qhat * FM, axis=1, keepdims=True)
    I = (np.abs(F) ** 2 + np.sum(np.abs(FM_perp) ** 2, axis=1)) / len(sc)
    return F, FM, I


def kernel_points(sc, sample):
    N = np.array(sample.N)
    xyz = np.array([site.xyz for site in sc.sites])
    return (
        sample.m.astype(np.int16),
        np.mod(sample.m, N).astype(np.int16),
        2 * np.pi * sc.crystal.B / N,
        *(np.exp(2j * np.pi * np.arange(n) / n) for n in N),
        np.exp(2j * np.pi * xyz @ sample.hkl.T),
    )


def kernel_structure_factors(sc, sample, nuc, mag, nuclear=True, magnetic=True):
    points = kernel_points(sc, sample)
    tpN = 2 * np.pi / np.array(sample.N, dtype=float)
    su = np.ascontiguousarray((sc.u @ np.linalg.inv(sc.crystal.A).T) * tpN)
    factors = (nuc.c, nuc.t, nuc.T, mag.c, mag.t, mag.T, nuclear, magnetic)
    F = np.zeros(len(sample), complex)
    FM = np.zeros((len(sample), 3), complex)
    kernels.structure_factors(
        points, factors, sc.cell.astype(np.int64), sc.site.astype(np.int64),
        su, sc.kind, sc.S, F, FM,
    )
    return points, F, FM


def test_supercell_composition(toy_crystal):
    sc = supercell(toy_crystal)
    assert len(sc) == np.prod(N) * 2
    fractions = sc.composition()
    for k, kind in enumerate(sc.kinds):
        assert abs(fractions[k] - kind.fraction) <= 1 / sc.n_cells
    assert {k.label for k in sc.kinds} == {"Fe1", "Mn1", "vacancy", "Co1"}


def test_structure_factors_match_brute_force(toy_crystal):
    sc = supercell(toy_crystal)
    sample = all_points(toy_crystal, N)
    debye_waller = [0]
    nuc = scatterers.nuclear_factors(sc, sample, debye_waller=debye_waller)
    mag = scatterers.magnetic_factors(sc, sample, debye_waller=debye_waller)
    assert nuc.T.shape[0] == 1 and mag.T.shape[0] == 3

    points, F, FM = kernel_structure_factors(sc, sample, nuc, mag)
    F0, FM0, I0 = brute_force(sc, sample, nuc.dense(), mag.dense())
    assert np.allclose(F, F0) and np.allclose(FM, FM0)

    I = np.empty(len(sample))
    kernels.intensity(points, F, FM, 1 / len(sc), True, I)
    assert np.allclose(I, I0)


def test_parseval_sum_rule(toy_crystal):
    """Over one reciprocal cell, sum |F_j|^2 = n_cells sum_i |b_i|^2 per site."""
    sc = supercell(toy_crystal)
    sc.u[:] = 0
    sample = all_points(toy_crystal, N, include_integer=True)
    nuc = scatterers.nuclear_factors(sc, sample)
    mag = scatterers.magnetic_factors(sc, sample)
    for site in range(sc.n_sites):
        c = nuc.c.copy()
        c[[k.site != site for k in sc.kinds]] = 0
        only = scatterers.Factors(c, nuc.t, nuc.T)
        _, F, _ = kernel_structure_factors(sc, sample, only, mag, magnetic=False)
        atoms = sc.atoms_on_site(site)
        expected = sc.n_cells * np.sum(np.abs(nuc.c[sc.kind[atoms]]) ** 2)
        assert np.isclose(np.sum(np.abs(F) ** 2), expected)


def test_fit_matches_least_squares():
    rng = np.random.default_rng(1)
    x = rng.random(50)
    y = 3 * x + 0.5 + 0.01 * rng.normal(size=50)
    w = rng.random(50) + 0.5
    chi, s, b = kernels.fit(
        w.sum(), (w * y).sum(), (w * y * y).sum(),
        (w * x).sum(), (w * x * x).sum(), (w * x * y).sum(),
        kernels.FIT_FREE, 1.0, 0.0,
    )
    A = np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    assert np.allclose([s, b], coef)
    assert np.isclose(chi, np.sum(w * (s * x + b - y) ** 2))


def test_site_penalty():
    rng = np.random.default_rng(2)
    U = np.array([[0.02, 0.005, 0], [0.005, 0.01, 0.0], [0, 0, 0.015]])
    L = np.linalg.cholesky(U)
    u = rng.normal(size=(400, 3)) @ L.T * 1.3 + 0.01
    Linv = np.linalg.inv(L)
    g = kernels.site_penalty(u.sum(0), u.T @ u, len(u), Linv)
    z = u @ Linv.T
    C = np.cov(z.T, bias=True)
    assert np.isclose(g, np.sum((C - np.eye(3)) ** 2) + np.sum(z.mean(0) ** 2))


@pytest.mark.parametrize(
    "magnetic, displacive", [(True, ()), (False, (1,)), (True, (1,)), (True, (0, 1))]
)
def test_incremental_updates_are_exact(toy_crystal, magnetic, displacive):
    sc = supercell(toy_crystal, rng=3)
    sample = all_points(toy_crystal, N)
    rng = np.random.default_rng(4)
    sample.y = rng.random(len(sample))
    sc.randomize_displacements()
    ref = Refinement(
        sc, sample, magnetic=magnetic, displacive=displacive,
        cone=60, step=0.5, seed=5,
        weights={"swap": 1, "spin": 1, "displace": 1, "swap_displace": 1, "exchange": 1},
    )
    ref.run(2000, temperature=1e3, decay=0.0)
    moves = ref.history.array("move")
    if 0 in displacive:
        assert np.any(moves == kernels.MOVE_SWAP_DISPLACE)
    if displacive:
        assert np.any(moves == kernels.MOVE_EXCHANGE)
    incremental = ref.su.copy(), ref.suu.copy(), ref.g.copy()
    ref.recompute()
    for a, b in zip(incremental, (ref.su, ref.suu, ref.g)):
        assert np.allclose(a, b, atol=1e-12)
    assert ref.drift < 1e-10
    assert np.isclose(ref.incremental_chi_sq, ref.chi_sq, rtol=1e-8)
    acc = ref.history.array("accepted")
    assert 0.2 < acc.mean() <= 1.0

    P, M = ref.P, ref.M
    _, _, I0 = brute_force(sc, sample, P, M)
    assert np.allclose(ref.intensity(), I0)

    for k in np.flatnonzero(ref.penalized):
        assert np.allclose(ref.su[k], sc.u[sc.kind == k].sum(0))
    assert not np.any(ref.penalized[[k.vacancy for k in sc.kinds]])
    assert np.all(np.linalg.norm(sc.S, axis=1) == pytest.approx(1.0))
    assert np.allclose(sc.composition(), [k.fraction for k in sc.kinds], atol=1 / sc.n_cells)


@pytest.mark.parametrize("magnetic", [False, True])
def test_nufft_matches_direct_displaced_sum(toy_crystal, magnetic):
    pytest.importorskip("finufft")
    sc = supercell(toy_crystal, rng=3)
    sample = all_points(toy_crystal, N)
    sc.randomize_displacements()
    ref = Refinement(sc, sample, magnetic=magnetic, displacive=(0, 1), seed=5)
    sel = np.flatnonzero(np.any(ref.su_scaled != 0, axis=1))
    assert sel.size > 0
    out = {}
    for method in ("direct", "nufft"):
        F = np.zeros(ref.n_points, dtype=complex)
        FM = np.zeros((ref.n_points, 3), dtype=complex)
        ref._displaced_sum(sel, F, FM, method=method)
        out[method] = F, FM
    scale = max(np.abs(out["direct"][0]).max(), np.abs(out["direct"][1]).max())
    for a, b in zip(out["direct"], out["nufft"]):
        assert np.abs(a - b).max() < 1e-10 * scale
    assert np.abs(out["direct"][0]).max() > 0
    if magnetic:
        assert np.abs(out["direct"][1]).max() > 0


def test_exchange_conserves_site_displacements(toy_crystal):
    sc = supercell(toy_crystal, rng=3)
    sc.randomize_displacements()
    before = [np.sort(sc.u[sc.atoms_on_site(j)], axis=0) for j in range(sc.n_sites)]
    sample = all_points(toy_crystal, N)
    sample.y = np.random.default_rng(4).random(len(sample))
    ref = Refinement(
        sc, sample, magnetic=True, displacive=(0, 1), seed=5,
        weights={"swap": 1, "spin": 1, "exchange": 1},
    )
    ref.run(2000, temperature=1e3)
    moves = ref.history.array("move")
    accepted = ref.history.array("accepted").astype(bool)
    assert np.any(accepted & (moves == kernels.MOVE_EXCHANGE))
    assert not np.any(moves == kernels.MOVE_DISPLACE)
    for j in range(sc.n_sites):
        after = np.sort(sc.u[sc.atoms_on_site(j)], axis=0)
        assert np.array_equal(after, before[j])
    incremental = ref.su.copy(), ref.suu.copy(), ref.g.copy()
    ref.recompute()
    for a, b in zip(incremental, (ref.su, ref.suu, ref.g)):
        assert np.allclose(a, b, atol=1e-12)
    assert ref.drift < 1e-10
    assert np.isclose(ref.incremental_chi_sq, ref.chi_sq, rtol=1e-8)


def test_refinement_recovers_antiferromagnetic_order():
    """Fit a staggered configuration with noise, starting from random spins."""

    from rmc_discord.crystal.cif import AsymmetricSite
    from rmc_discord.crystal.structure import Crystal

    crystal = Crystal(
        (4.0, 4.0, 4.0, 90, 90, 90), "P 1",
        [AsymmetricSite("Mn1", "Mn", np.zeros(3), 1.0, U_iso=0.0)],
    )
    Ns = np.array((6, 6, 6))

    def make(rng):
        return Supercell(crystal, Ns, ["Mn1"], ions={"Mn1": "Mn2+"},
                         moments={"Mn2+": 5.0}, rng=rng)

    truth = make(0)
    sign = np.where(np.sum(truth.cell, axis=1) % 2 == 0, 1.0, -1.0)
    noise = np.random.default_rng(1).normal(scale=0.6, size=(len(truth), 3))
    S = sign[:, None] * np.array([0.0, 0.0, 1.0]) + noise
    truth.S = S / np.linalg.norm(S, axis=1, keepdims=True)

    m = np.stack(np.meshgrid(*[np.arange(-n, n) for n in Ns], indexing="ij"), -1)
    m = m.reshape(-1, 3)
    m = m[~np.all(m % Ns == 0, axis=1)]
    hkl = m / Ns
    sample = QSample(
        m=m, N=tuple(Ns), hkl=hkl, Q=crystal.Q(hkl), y=np.ones(len(m)),
        variance=np.ones(len(m)), bins=np.ones(len(m), dtype=int),
    )
    P = scatterers.nuclear_prefactor(truth, sample)
    M = scatterers.magnetic_prefactor(truth, sample)
    _, _, I_true = brute_force(truth, sample, P, M)
    sample.y = 2.0 * I_true + 0.1
    sample.variance = 1e-3 * sample.y.mean() * sample.y + 1e-6

    def staggered(S):
        return np.linalg.norm(np.mean(sign[:, None] * S, axis=0))

    sc = make(2)
    ref = Refinement(sc, sample, nuclear=False, magnetic=True, cone=30, seed=3)
    chi0 = ref.reduced_chi_sq
    ref.run(len(sc) * 5, temperature=1.0, decay=1e-4)
    for _ in range(29):
        ref.run(len(sc) * 5, decay=1e-4)

    assert ref.reduced_chi_sq < 0.1 * chi0
    assert np.isclose(ref.scale, 2.0, rtol=0.05)
    assert np.isclose(staggered(sc.S), staggered(truth.S), atol=0.05)


def test_refinement_recovers_displacement_correlations():
    """Anticorrelated nearest-neighbor displacements at large Q.u."""

    from rmc_discord.crystal.cif import AsymmetricSite
    from rmc_discord.crystal.structure import Crystal

    U_iso = 0.02
    crystal = Crystal(
        (3.0, 3.0, 3.0, 90, 90, 90), "P 1",
        [AsymmetricSite("Pb1", "Pb", np.zeros(3), 1.0, U_iso=U_iso)],
    )
    Ns = np.array((6, 6, 6))

    def make(rng):
        return Supercell(crystal, Ns, ["Pb1"], rng=rng)

    truth = make(0)
    sign = np.where(truth.cell[:, 0] % 2 == 0, 1.0, -1.0)
    z = np.random.default_rng(1).normal(size=(len(truth), 3))
    z[:, 0] = 0.8 * sign + 0.6 * z[:, 0]
    truth.u = np.sqrt(U_iso) * z

    m = np.stack(np.meshgrid(*[np.arange(0, 3 * n) for n in Ns], indexing="ij"), -1)
    m = m.reshape(-1, 3)
    m = m[~np.all(m % Ns == 0, axis=1)]
    hkl = m / Ns
    sample = QSample(
        m=m, N=tuple(Ns), hkl=hkl, Q=crystal.Q(hkl), y=np.ones(len(m)),
        variance=np.ones(len(m)), bins=np.ones(len(m), dtype=int),
    )
    assert np.max(sample.Q_norm) * np.sqrt(U_iso) * 3 > 1

    P = scatterers.nuclear_prefactor(truth, sample)
    _, _, I_true = brute_force(truth, sample, P, np.zeros(P.shape))
    sample.y = 3.0 * I_true + 0.05
    sample.variance = 1e-3 * sample.y.mean() * sample.y + 1e-6

    sc = make(2)
    sc.randomize_displacements()
    ref = Refinement(sc, sample, displacive=[0], step=0.5, penalty=1000.0, seed=3)
    chi0 = ref.reduced_chi_sq
    ref.run(len(sc) * 5, temperature=1.0, decay=1e-4, recompute_every=len(sc), adapt=0.3)
    for _ in range(29):
        ref.run(len(sc) * 5, decay=1e-4, recompute_every=len(sc), adapt=0.3)

    assert ref.reduced_chi_sq < 0.2 * chi0
    assert np.isclose(ref.scale, 3.0, rtol=0.3)

    C = np.cov(sc.u.T, bias=True)
    assert np.allclose(np.diag(C), U_iso, rtol=0.1)

    u = sc.u.reshape(*Ns, 3)
    u_true = truth.u.reshape(*Ns, 3)
    nn = np.mean(u[..., 0] * np.roll(u, 1, axis=0)[..., 0]) / U_iso
    nn_true = np.mean(u_true[..., 0] * np.roll(u_true, 1, axis=0)[..., 0]) / U_iso
    assert nn < -0.3 and np.isclose(nn, nn_true, atol=0.15)


def test_fft_recompute_matches_brute_force(toy_crystal):
    """Undisplaced atoms by FFT, displaced ones directly, with tables."""
    sc = supercell(toy_crystal, rng=6)
    rng = np.random.default_rng(7)
    still = rng.random(len(sc)) < 0.6
    sc.u[still] = 0
    sample = all_points(toy_crystal, N)
    ref = Refinement(sc, sample, magnetic=True, displacive=[1], seed=8)
    assert np.any(~np.any(ref.su_scaled != 0, axis=1))
    assert np.any(np.any(ref.su_scaled != 0, axis=1))
    F0, FM0, I0 = brute_force(sc, sample, ref.P, ref.M)
    assert np.allclose(ref.F, F0) and np.allclose(ref.FM, FM0)
    assert np.allclose(ref.intensity(), I0)


def test_minibatch_subsets(toy_crystal):
    sc = supercell(toy_crystal, rng=9)
    sample = all_points(toy_crystal, N)
    sample.y = np.random.default_rng(10).random(len(sample))
    ref = Refinement(sc, sample, magnetic=True, displacive=[1], seed=11)
    ref.run(600, temperature=1e2, recompute_every=200, subset=0.4, restore=False)
    assert len(ref.sample) < len(sample)
    incremental = ref.chi_sq
    ref.recompute()
    assert ref.drift < 1e-10
    assert np.isclose(incremental, ref.chi_sq, rtol=1e-8)
    ref.run(600, recompute_every=200, subset=0.4)
    assert len(ref.sample) == len(sample)
    F0, FM0, I0 = brute_force(sc, sample, ref.P, ref.M)
    assert np.allclose(ref.intensity(), I0)


def carbonate_supercell(crystal, N=(2, 2, 3), rng=0):
    return Supercell(crystal, N, ["C2", "Ba1"], groups={"C2": ["O8"]}, rng=rng)


def test_rigid_groups_from_symmetry(carbonate_crystal):
    sc = carbonate_supercell(carbonate_crystal)
    groups = [k for k in sc.kinds if k.group]
    assert len(groups) == 2
    assert {k.symbol for k in groups} == {"CO3"}
    for k in groups:
        d = np.array([m.d for m in k.members])
        assert np.allclose(d[0], 0)
        r = np.linalg.norm(d[1:], axis=1)
        assert np.allclose(r, r[0]) and 1.2 < r[0] < 1.35
        assert np.allclose(d[1:, 2], 0)
        assert np.isclose(k.fraction, 0.8)
    assert sum(k.vacancy for k in sc.kinds) == 2

    r, labels, b, owner = sc.atoms()
    n_groups = np.sum([sc.kinds[k].group for k in sc.kind])
    assert np.sum(labels == "O8") == 3 * n_groups
    assert np.sum(labels == "Ba1") == 3 * sc.n_cells


def test_rigid_group_structure_factors_match_atoms(carbonate_crystal):
    sc = carbonate_supercell(carbonate_crystal, rng=1)
    sc.randomize_displacements()
    half = sc.rng.random(len(sc)) < 0.5
    sc.u[half] = 0.0
    sample = all_points(carbonate_crystal, sc.N)
    sample.y = np.random.default_rng(2).random(len(sample))
    ref = Refinement(sc, sample, displacive=range(sc.n_sites), step=0.5, seed=3)

    def atoms_F():
        r, _, b, _ = sc.atoms()
        return np.exp(1j * sample.Q @ r.T) @ b

    assert np.allclose(ref.F, atoms_F())

    ref.run(3000, temperature=1e3)
    moves = ref.history.array("move")
    assert np.any((moves == kernels.MOVE_SWAP_DISPLACE) & (ref.history.array("accepted") == 1))
    assert np.allclose(ref.F, atoms_F())
    ref.recompute()
    assert ref.drift < 1e-10


def channel_neighbors(sc):
    """Next and previous group along c in each carbonate channel."""
    z = sc.r[:, 2]
    column = sc.cell[:, 0] * sc.N[1] + sc.cell[:, 1]
    up = np.empty(len(sc), int)
    down = np.empty(len(sc), int)
    for c in np.unique(column):
        atoms = np.flatnonzero(column == c)
        atoms = atoms[np.argsort(z[atoms])]
        up[atoms] = np.roll(atoms, -1)
        down[atoms] = np.roll(atoms, 1)
    return up, down


def size_effect(sc, up, down):
    """<(v_up - v_down) u_z> over present groups, v = 1 if vacant."""
    v = np.array([sc.kinds[k].vacancy for k in sc.kind], dtype=float)
    present = v == 0
    return np.mean((v[up] - v[down])[present] * sc.u[present, 2])


@pytest.mark.parametrize("delta", [0.15, 0.0])
def test_refinement_recovers_size_effect(delta):
    """Groups next to a vacancy along c shift toward it by ``delta`` (A)."""

    from rmc_discord.crystal.cif import AsymmetricSite
    from rmc_discord.crystal.structure import Crystal

    crystal = Crystal(
        (9.683, 9.683, 9.518, 90, 90, 120), "P -6",
        [
            AsymmetricSite("C2", "C", np.array([1 / 3, 2 / 3, 0.187]), 0.8, U_iso=0.008),
            AsymmetricSite("O8", "O", np.array([0.203, 0.529, 0.187]), 0.8, U_iso=0.008),
        ],
    )
    n = (3, 3, 8)

    def group_supercell(seed):
        return Supercell(crystal, n, ["C2"], groups={"C2": ["O8"]}, rng=seed)

    up, down = channel_neighbors(group_supercell(0))
    truths = []
    for seed in range(8):
        t = group_supercell(seed)
        t.randomize_displacements(scale=0.5)
        v = np.array([t.kinds[k].vacancy for k in t.kind], dtype=float)
        t.u[:, 2] += delta * (v[up] - v[down])
        t.u[v == 1] = 0
        truths.append(t)
    expected = np.mean([size_effect(t, up, down) for t in truths])
    U = np.mean([np.cov(t.u[t.kind % 2 == 0].T, bias=True) for t in truths], axis=0)

    rng = np.random.default_rng(1)
    N = np.array(n)
    axes = [np.arange(-3 * N[0], 3 * N[0] + 1)] * 2 + [np.arange(-6 * N[2], 6 * N[2] + 1)]
    m = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    m = m[~np.all(m % N == 0, axis=1)]
    m = m[rng.random(len(m)) < 12000 / len(m)]
    hkl = m / N
    sample = QSample(
        m=m, N=n, hkl=hkl, Q=crystal.Q(hkl), y=np.zeros(len(m)),
        variance=np.ones(len(m)), bins=np.ones(len(m), int),
    )

    def with_U(sc):
        for k in sc.kinds:
            if k.group:
                k.U = U
        return sc

    I = np.mean(
        [Refinement(with_U(t), sample, displacive=[0, 1], fit="none").intensity()
         for t in truths],
        axis=0,
    )
    sample.y = I * (1 + 0.02 * rng.standard_normal(len(I)))
    sample.variance = (0.02 * I + 0.01 * I.mean()) ** 2

    sc = with_U(group_supercell(5))
    sc.randomize_displacements()
    ref = Refinement(sc, sample, displacive=[0, 1], penalty=100.0, step=0.3,
                     fit="none", seed=2)
    ref.calibrate_temperature()
    for _ in range(40):
        ref.run(4 * len(sc), recompute_every=len(sc), adapt=0.3)
        ref.temperature *= 0.88

    found = size_effect(sc, up, down)
    if delta == 0:
        assert abs(found) < 0.01
    else:
        assert 0.6 * expected < found < 1.2 * expected
    assert ref.penalty_value() < 0.2 * ref.penalty * 3
    assert max(ref.drifts) < 1e-10


def test_radial_background_matches_least_squares(toy_crystal):
    sc = supercell(toy_crystal, rng=6)
    sample = all_points(toy_crystal, (6, 6, 4))
    sc = Supercell(toy_crystal, (6, 6, 4), ["Fe1", "Co1"], rng=6)
    sc.randomize_displacements()
    rng = np.random.default_rng(7)
    Q = sample.Q_norm
    sample.y = rng.random(len(sample)) + 0.3 * np.sin(Q)
    sample.variance = rng.random(len(sample)) * 0.1 + 0.05
    ref = Refinement(sc, sample, displacive=[1], background_basis="radial",
                     knot_spacing=0.4, seed=8)

    def lstsq():
        K = len(ref.knots)
        B = np.zeros((len(sample), K))
        B[np.arange(len(sample)), ref.kb] = 1 - ref.tb
        B[np.arange(len(sample)), ref.kb + 1] += ref.tb
        X = np.column_stack([ref.intensity() - ref.offset, B])
        sw = np.sqrt(ref.w)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], sample.y * sw, rcond=None)
        if coef[0] < 0:  # the scale is constrained to be nonnegative
            X[:, 0] = 0
            coef, *_ = np.linalg.lstsq(X * sw[:, None], sample.y * sw, rcond=None)
        return coef, X @ coef

    coef, fitted = lstsq()
    assert coef[0] > 0
    assert np.isclose(ref.scale, coef[0], rtol=1e-6)
    assert np.isclose(ref.chi_sq, np.sum(ref.w * (sample.y - fitted) ** 2), rtol=1e-8)
    assert np.allclose(ref.model(), fitted, atol=1e-8)

    ref.run(2000, temperature=1e2)
    incremental = ref.chi_sq
    ref.recompute()
    assert np.isclose(incremental, ref.chi_sq, rtol=1e-8)
    coef, fitted = lstsq()
    assert np.isclose(ref.chi_sq, np.sum(ref.w * (sample.y - fitted) ** 2), rtol=1e-8)


def test_calibration_with_zero_scale(toy_crystal):
    """Uncorrelated start (scale clamped to 0) still gives a usable temperature."""
    sc = Supercell(toy_crystal, (6, 6, 4), ["Fe1", "Co1"], rng=1)
    sample = all_points(toy_crystal, (6, 6, 4))
    ref = Refinement(sc, sample, seed=2)
    ref.scale = 0.0
    ref.full_sample.y[:] = 1.0 - 1e-3 * ref.intensity()
    ref._set_points(np.arange(len(sample)))
    assert ref.scale == 0.0
    T = ref.calibrate_temperature()
    assert ref.fit_mode == kernels.FIT_FREE
    assert T > 1e-8


@pytest.mark.parametrize("basis", [None, "radial"])
def test_symmetrized_model_averages_equivalents(carbonate_crystal, basis):
    from rmc_discord.engine.refinement import parse_operator

    assert parse_operator("k,h,l").tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert parse_operator("-h+k,-h,l").tolist() == [[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]

    sc = carbonate_supercell(carbonate_crystal, N=(3, 3, 2), rng=4)
    sc.randomize_displacements()
    sample = all_points(carbonate_crystal, sc.N)
    sample.y = np.random.default_rng(5).random(len(sample)) + 1
    ref = Refinement(sc, sample, displacive=range(sc.n_sites), symmetrize=["k,h,l"],
                     background_basis=basis, seed=6)
    assert ref.n_points == 2 * len(sample)

    mirrored = all_points(carbonate_crystal, sc.N)
    mirrored.m = sample.m[:, [1, 0, 2]]
    mirrored.hkl = sample.hkl[:, [1, 0, 2]]
    mirrored.Q = carbonate_crystal.Q(mirrored.hkl)
    I1 = Refinement(sc, sample, displacive=range(sc.n_sites), fit="none").intensity()
    I2 = Refinement(sc, mirrored, displacive=range(sc.n_sites), fit="none").intensity()
    assert not np.allclose(I1, I2)
    assert np.allclose(ref.intensity(), 0.5 * (I1 + I2))

    ref.run(3000, temperature=1e2)
    incremental = ref.chi_sq
    ref.recompute()
    assert ref.drift < 1e-10
    assert np.isclose(incremental, ref.chi_sq, rtol=1e-8)
    r = sample.y - ref.model()
    assert np.isclose(ref.chi_sq, np.sum(ref.w * r**2), rtol=1e-8)

    with pytest.raises(ValueError):
        Refinement(carbonate_supercell(carbonate_crystal, N=(3, 2, 2)),
                   all_points(carbonate_crystal, (3, 2, 2)), symmetrize=["k,h,l"])


def test_anneal_with_held_out_validation(tmp_path):
    """Minibatch annealing improves held-out chi^2 and keeps the best state."""

    from rmc_discord.crystal.cif import AsymmetricSite
    from rmc_discord.crystal.structure import Crystal
    from rmc_discord.engine.anneal import Schedule, anneal, split_sample

    crystal = Crystal(
        (4.0, 4.0, 4.0, 90, 90, 90), "P 1",
        [AsymmetricSite("Mn1", "Mn", np.zeros(3), 1.0, U_iso=0.0)],
    )
    Ns = np.array((6, 6, 6))

    def make(rng):
        return Supercell(crystal, Ns, ["Mn1"], ions={"Mn1": "Mn2+"}, moments={"Mn2+": 5.0}, rng=rng)

    truth = make(0)
    sign = np.where(np.sum(truth.cell, axis=1) % 2 == 0, 1.0, -1.0)
    S = sign[:, None] * np.array([0.0, 0.0, 1.0]) + np.random.default_rng(1).normal(scale=0.6, size=(len(truth), 3))
    truth.S = S / np.linalg.norm(S, axis=1, keepdims=True)
    m = np.stack(np.meshgrid(*[np.arange(-n, n) for n in Ns], indexing="ij"), -1).reshape(-1, 3)
    m = m[~np.all(m % Ns == 0, axis=1)]
    hkl = m / Ns
    sample = QSample(m=m, N=tuple(Ns), hkl=hkl, Q=crystal.Q(hkl), y=np.ones(len(m)),
                     variance=np.ones(len(m)), bins=np.ones(len(m), dtype=int))
    P = scatterers.nuclear_prefactor(truth, sample)
    M = scatterers.magnetic_prefactor(truth, sample)
    _, _, I_true = brute_force(truth, sample, P, M)
    sample.y = 2.0 * I_true + 0.1
    sample.variance = 1e-3 * sample.y.mean() * sample.y + 1e-6

    Q = sample.Q_norm
    strata = {"low": Q < np.median(Q), "high": Q >= np.median(Q)}
    pool, held = split_sample(sample, held_out=0.2, strata=strata, seed=0)
    assert set(held) == {"low", "high"}
    assert len(pool) + sum(len(h) for h in held.values()) == len(sample)

    sc = make(2)
    ref = Refinement(sc, sample.select(pool), nuclear=False, magnetic=True, cone=30, seed=3)
    lines = []
    schedule = Schedule(rounds=15, sweeps=10, cool=0.8, temperature=1.0, batch=len(pool) // 2,
                        batch_sweeps=5, check_every=3)
    out = anneal(ref, held, schedule, save=tmp_path / "best.npz", log=lines.append)
    assert len(out.records) == 5 and len(lines) == 8  # baseline, threads, start T, 5 checks
    assert out.best["score"] == min(r["score"] for r in out.records)
    assert out.best["score"] < 0.2
    assert np.array_equal(sc.S, out.best["S"])
    assert np.allclose(np.load(tmp_path / "best.npz")["S"], out.best["S"])
    best = out.records[[r["round"] for r in out.records].index(out.best["round"])]
    assert ref.score(held["low"], scale=best["scale"]) == pytest.approx(best["held"]["low"])

    calls, rounds = [], []
    sc2 = make(2)
    ref2 = Refinement(sc2, sample.select(pool), nuclear=False, magnetic=True, cone=30, seed=3)

    def stop():
        rounds.append(1)
        return len(rounds) == 4

    out2 = anneal(ref2, held, schedule, log=None, callback=lambda row, res: calls.append(row["round"]), stop=stop)
    assert calls == [2, 3] and [r["round"] for r in out2.records] == calls


@pytest.mark.parametrize("basis", [None, "radial"])
def test_fused_nuclear_path_matches_general(carbonate_crystal, basis):
    """Phase-table kernels give the same chain as the general kernels."""
    chains = []
    for fused in (True, False):
        sc = carbonate_supercell(carbonate_crystal, N=(3, 3, 2), rng=4)
        sc.randomize_displacements()
        sample = all_points(carbonate_crystal, sc.N)
        sample.y = np.random.default_rng(5).random(len(sample)) + 1
        ref = Refinement(sc, sample, displacive=range(sc.n_sites), symmetrize=["k,h,l"], background_basis=basis,
                         penalty=1.0, seed=6,
                         weights={"swap": 1, "displace": 1, "swap_displace": 1, "exchange": 1})
        assert ref.use_fused
        ref.use_fused = fused
        ref.run(1500, temperature=1e2)
        chains.append((ref.history.array("chi_sq"), ref.history.array("move"), ref.F.copy(), sc.kind.copy()))
        if fused:
            incremental = ref.chi_sq
            ref.recompute()
            assert ref.drift < 1e-10
            assert np.isclose(incremental, ref.chi_sq, rtol=1e-8)
    (c1, m1, F1, k1), (c2, m2, F2, k2) = chains
    assert {0, 2, 3, 4} <= set(np.unique(m1).tolist())
    assert np.array_equal(m1, m2) and np.array_equal(k1, k2)
    assert np.allclose(c1, c2, rtol=1e-9)
    assert np.allclose(F1, F2, rtol=1e-9, atol=1e-9 * np.abs(F2).max())


@pytest.mark.parametrize("nuclear", [True, False])
@pytest.mark.parametrize("offset", [None, "paramagnetic"])
def test_fused_magnetic_path_matches_general(toy_crystal, nuclear, offset):
    chains = []
    for fused in (True, False):
        sc = supercell(toy_crystal, rng=3)
        sc.randomize_displacements()
        sample = all_points(toy_crystal, N)
        sample.y = np.random.default_rng(4).random(len(sample))
        ref = Refinement(sc, sample, magnetic=True, nuclear=nuclear, displacive=[0, 1], offset=offset,
                         cone=60, step=0.5, seed=5, background_basis="radial",
                         weights={"swap": 1, "spin": 1, "displace": 1, "swap_displace": 1, "exchange": 1})
        ref.use_fused = fused
        ref.run(2000, temperature=1e2)
        chains.append((ref.history.array("chi_sq"), ref.history.array("move"), ref.F.copy(), ref.FM.copy()))
        if fused:
            incremental = ref.chi_sq
            ref.recompute()
            assert ref.drift < 1e-10
            assert np.isclose(incremental, ref.chi_sq, rtol=1e-8)
    (c1, m1, F1, M1), (c2, m2, F2, M2) = chains
    assert set(np.unique(m1).tolist()) == {0, 1, 2, 3, 4}
    assert np.array_equal(m1, m2)
    assert np.allclose(c1, c2, rtol=1e-9)
    assert np.allclose(F1, F2, atol=1e-9 * max(np.abs(F2).max(), 1))
    assert np.allclose(M1, M2, atol=1e-9 * np.abs(M2).max())


def test_moves_that_cannot_change_the_intensity_are_rejected(toy_crystal):
    """Swaps without the nuclear term only count on sites with a moment."""
    sample = all_points(toy_crystal, N)
    with pytest.raises(ValueError, match="neither nuclear nor magnetic"):
        Refinement(supercell(toy_crystal), sample, nuclear=False, magnetic=False)
    # Fe1/Mn1 are both magnetic: swaps change the magnetic intensity
    ref = Refinement(supercell(toy_crystal), sample, nuclear=False, magnetic=True, weights={"swap": 1})
    assert ref.swap_sites.size == 1
    # without moments a swap is invisible to the magnetic term
    sc = Supercell(toy_crystal, N, ["Fe1"], rng=0)
    with pytest.raises(ValueError, match="swaps need the nuclear term"):
        Refinement(sc, sample, nuclear=False, magnetic=True, weights={"swap": 1})


def test_thread_tuning_restores_the_refinement(toy_crystal):
    """Timing bursts leave the configuration, structure factors, and fit as they were."""
    import numba

    sc = supercell(toy_crystal, rng=3)
    sample = all_points(toy_crystal, N)
    sample.y = np.random.default_rng(4).random(len(sample))
    sc.randomize_displacements()
    ref = Refinement(sc, sample, magnetic=True, displacive=[0], cone=60, seed=5,
                     weights={"swap": 1, "spin": 1, "displace": 1})
    ref.run(200, temperature=1.0)
    before = (sc.kind.copy(), sc.u.copy(), sc.S.copy(), ref.F.copy(), ref.FM.copy(), ref.su_scaled.copy(),
              ref.chi_sq, ref.scale, ref.temperature, len(ref.history.array("move")))
    threads = ref.tune_threads(candidates=[1, 2], n_moves=50, repeats=1)
    assert threads in (1, 2) and numba.get_num_threads() == threads
    after = (sc.kind, sc.u, sc.S, ref.F, ref.FM, ref.su_scaled,
             ref.chi_sq, ref.scale, ref.temperature, len(ref.history.array("move")))
    for a, b in zip(before, after):
        assert np.array_equal(a, b)
    ref.recompute()
    assert ref.drift < 1e-12
    ref.run(200, temperature=1.0)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
