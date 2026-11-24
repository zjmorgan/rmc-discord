import numpy as np
import pytest

from rmc_discord.pairs.constraints import stack_parameterizations
from rmc_discord.pairs.sampling import Observations
from rmc_discord.pairs.workflow import ModelSpec, PairModel


def observations(h, y, sigma, held):
    m = len(h)
    return Observations(h, y, sigma, np.zeros((m, 3), int), np.ones(m, int), np.full(3, 0.05), np.eye(3), held)


def synthetic(model, rng, m=1500, scale=1.3, noise=1e-3):
    """Data from coefficients satisfying the model's constraints."""
    par = stack_parameterizations(*[b.parameterization for b in model.blocks])
    theta = 0.05 * rng.normal(size=par.n_free)
    c = par.c0 + par.N @ theta
    h = rng.uniform(-4, 4, (m, 3))
    obs = observations(h, np.zeros(m), np.ones(m), rng.random(m) < 0.2)
    A = model.design(h, obs.widths, obs.projection)
    B = model.background(obs, "polynomial", degree=1)(h)
    y0 = scale * A @ c + B @ np.array([0.3, 0.05]) * np.abs(A @ c).mean()
    sigma = noise * (np.abs(y0) + np.abs(y0).mean())
    obs.y, obs.sigma = y0 + sigma * rng.normal(size=m), sigma
    return obs, c


def test_occupancy_fit_recovers_coefficients_and_scale(tetragonal):
    model = PairModel(tetragonal, ModelSpec(r_max=6.0, cluster_distance=1.0, collapse=False))
    assert [b.name for b in model.blocks] == ["occupancy"]
    obs, c = synthetic(model, np.random.default_rng(0))
    fit = model.fit(obs, background="polynomial", degree=1)
    assert np.isclose(fit.refinement.scale, 1.3, rtol=1e-3)
    assert np.allclose(fit.refinement.c, c, atol=5 * fit.refinement.esd.max() + 1e-6)
    assert 0.5 < fit.chi2_test < 2.0 and fit.n_test == obs.held.sum()
    table = model.tables(fit)["occupancy"]
    o = model.blocks[0].orbits
    assert np.allclose(table.value, fit.refinement.c[o.orbit])
    assert np.allclose(model.predict(fit, obs.h[:5], obs.widths, obs.projection),
                       fit.refinement.predict(model.design(obs.h[:5], obs.widths, obs.projection),
                                              fit.background(obs.h[:5])))


@pytest.mark.parametrize("difference", [False, True])
def test_isotropic_magnetic_fit(tetragonal, difference):
    spec = ModelSpec(active=[], r_max=7.0, magnetic="isotropic", magnetic_sites=["M1"], ions={"M1": "Ni2+"},
                     moments={"Ni2+": 2.0}, difference=difference)
    model = PairModel(tetragonal, spec)
    block = model.blocks[0]
    assert block.name == "spin"
    assert (len(block.columns) < block.orbits.n_orbits) == difference
    obs, c = synthetic(model, np.random.default_rng(1), scale=1.0 if difference else 1.3)
    fit = model.fit(obs, background="polynomial", degree=1)
    assert np.allclose(fit.refinement.c, c, atol=5 * fit.refinement.esd.max() + 1e-6)
    table = model.tables(fit)["spin"]
    self_pairs = table.distance < 1e-8
    assert np.all(self_pairs == False) if difference else np.allclose(table.value[self_pairs], 1.0)


def test_structural_channels_build(tetragonal):
    spec = ModelSpec(r_max=5.0, cluster_distance=1.0, size_effect=True, displacement=True)
    model = PairModel(tetragonal, spec)
    assert [b.name for b in model.blocks] == ["occupancy", "size effect", "displacement"]
    h = np.random.default_rng(2).uniform(-3, 3, (20, 3))
    assert model.design(h).shape == (20, model.n_coefficients)


def test_identifiability_ties_join_indistinguishable_orbits(tetragonal):
    from rmc_discord.pairs.normal import NormalEquations

    model = PairModel(tetragonal, ModelSpec(r_max=6.0, cluster_distance=1.0, collapse=False))
    block = model.blocks[0]
    free = [k for k in range(block.n) if k not in set(block.fixed[0])]
    free = [k for k in free if np.any(np.abs(block.parameterization.N[k]) > 1e-10)]
    a, b = free[-2], free[-1]
    X = np.random.default_rng(3).normal(size=(200, block.n))
    X[:, b] = -2 * X[:, a]  # same column up to scale (and sign): not separable
    neq = NormalEquations(block.n).add(X, np.zeros(200), np.ones(200))
    parts, n_ties = model.identifiability_ties(neq)
    assert n_ties == 1
    N = parts[0].N
    assert np.allclose(N[b], -2 * N[a]) and parts[0].n_free == block.parameterization.n_free - 1
    _, none = model.identifiability_ties(neq, tol=0)
    assert none == 0


def test_identifiability_ties_respect_existing_ties(tetragonal):
    # a data tie c_b = 2 c_a on orbits already tied c_a = c_b would force both to zero
    from rmc_discord.pairs.constraints import parameterize
    from rmc_discord.pairs.normal import NormalEquations

    model = PairModel(tetragonal, ModelSpec(r_max=6.0, cluster_distance=1.0))
    block = model.blocks[0]
    free = [k for k in range(block.n) if k not in set(block.fixed[0])]
    free = [k for k in free if np.any(np.abs(block.parameterization.N[k]) > 1e-10)]
    a, b = free[-2], free[-1]
    row = np.zeros(block.n)
    row[a], row[b] = 1, -1
    block.rows = row[None] if block.rows is None else np.vstack([block.rows, row])
    block.parameterization = parameterize(block.n, block.fixed, block.rows)
    X = np.random.default_rng(3).normal(size=(200, block.n))
    X[:, b] = 2 * X[:, a]
    neq = NormalEquations(block.n).add(X, np.zeros(200), np.ones(200))
    parts, n_ties = model.identifiability_ties(neq)
    assert n_ties == 0
    N = parts[0].N
    assert np.linalg.norm(N[a]) > 1e-8 and np.allclose(N[a], N[b])


def test_collapsed_groups_recover_weighted_means(tetragonal):
    # without Debye-Waller factors, orbits with the same pair vectors have proportional columns
    # (b_i b_j): the data fix only the weighted group sums, which the collapsed fit recovers
    from rmc_discord.pairs.workflow import group_means

    full = PairModel(tetragonal, ModelSpec(r_max=6.0, cluster_distance=1.0, thermal=False, collapse=False))
    model = PairModel(tetragonal, ModelSpec(r_max=6.0, cluster_distance=1.0, thermal=False))
    block = model.blocks[0]
    assert block.groups is not None and len(np.unique(block.groups)) < block.n
    assert model.n_free < full.n_free
    obs, c = synthetic(full, np.random.default_rng(4))
    fit = model.fit(obs, background="polynomial", degree=1)
    ref = full.fit(obs, background="polynomial", degree=1)
    assert np.isclose(fit.chi2_train, ref.chi2_train, rtol=1e-6)  # nothing the data see is lost
    assert np.isclose(fit.refinement.scale, 1.3, rtol=1e-3)
    truth, _ = group_means(block.groups, block.weights, c, np.zeros_like(c))
    got, err = group_means(block.groups, block.weights, fit.refinement.c, fit.refinement.esd, fit.refinement.cov)
    free = np.flatnonzero(np.linalg.norm(block.parameterization.N, axis=1) > 1e-10)
    assert np.allclose(got[free], truth[free], atol=5 * err[free].max() + 1e-6)
    table = model.tables(fit)["occupancy"]
    o = block.orbits
    assert np.allclose(table.value, got[o.orbit])


def test_collapse_keeps_constraint_separated_members():
    from rmc_discord.pairs.constraints import collapse_parameterization, parameterize

    # c0 + c1 tied in one group, but c1 = 2 c2 links c1 to an independent group: stays resolvable
    par = parameterize(3, None, np.array([[0.0, 1.0, -2.0]]))
    out = collapse_parameterization(par, np.array([0, 0, 2]), np.array([1.0, 1.0, 1.0]))
    assert out.n_free == 2
    # a group with no outside link keeps only its sum
    out = collapse_parameterization(parameterize(3), np.array([0, 0, 2]), np.array([1.0, 2.0, 1.0]))
    assert out.n_free == 2
    N = out.N
    assert np.allclose(N[0] * 2, N[1])  # least-norm split along the weights


def test_displacement_plane_and_axis(tetragonal):
    """In-plane displacements: v . n = 0 and V n = 0 for (001); an axis leaves only u u^T."""
    st = tetragonal
    full = PairModel(st, ModelSpec(r_max=6.0, cluster_distance=1.0, size_effect=True, displacement=True))
    layer = PairModel(st, ModelSpec(r_max=6.0, cluster_distance=1.0, size_effect=True, displacement=True,
                                    plane=(0, 0, 1)))
    chain = PairModel(st, ModelSpec(r_max=6.0, cluster_distance=1.0, size_effect=True, displacement=True,
                                    axis=(0, 0, 1)))
    n = st.B @ [0, 0, 1]
    n = n / np.linalg.norm(n)
    blocks = {b.name: b for b in layer.blocks}
    for v in blocks["size effect"].fields.vectors:
        assert np.allclose(v @ n, 0)
    for t in blocks["displacement"].fields.tensors:
        assert np.allclose(t @ n, 0)
    sizes = [{b.name: b.n for b in m.blocks} for m in (full, layer, chain)]
    for name in ("size effect", "displacement"):
        assert sizes[0][name] > sizes[1][name] > 0 and sizes[0][name] > sizes[2][name] > 0
