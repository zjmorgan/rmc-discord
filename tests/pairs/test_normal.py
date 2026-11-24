import numpy as np
import pytest

from rmc_discord.pairs.constraints import parameterize
from rmc_discord.pairs.normal import NormalEquations, accumulate, held_out_chi2, refine_normal
from rmc_discord.pairs.refine import background_basis, refine

from test_refine import problem


@pytest.mark.parametrize("ridge", [0.0, 0.3])
def test_streaming_equals_dense_refinement(ridge):
    rng = np.random.default_rng(7)
    A, c, fixed, B, beta = problem(rng)
    sigma = rng.uniform(0.5, 1.5, len(A))
    y = 2.5 * A @ c + B @ beta + 0.2 * sigma * rng.normal(size=len(A))
    rows = np.zeros((1, A.shape[1]))
    rows[0, [1, 2]] = [1, -1]  # tie two coefficients
    par = parameterize(A.shape[1], fixed, rows)
    dense = refine(y, sigma, A, B=B, parameterization=par, ridge=ridge)
    neq = NormalEquations(A.shape[1], B.shape[1])
    for s in np.array_split(np.arange(len(A)), 7):  # tiles
        neq.add(A[s], y[s], sigma[s], B[s])
    stream = refine_normal(neq, parameterization=par, ridge=ridge)
    for key in ("c", "esd", "background", "background_esd"):
        assert np.allclose(getattr(stream, key), getattr(dense, key), rtol=1e-7, atol=1e-9), key
    assert np.isclose(stream.scale, dense.scale) and np.isclose(stream.scale_esd, dense.scale_esd)
    assert np.isclose(stream.chi2, np.sum(((y - dense.predict(A, B)) / sigma) ** 2), rtol=1e-8)
    assert stream.rank == dense.rank and stream.n_params == dense.n_params


def test_accumulate_splits_and_held_out_chi2():
    rng = np.random.default_rng(3)
    A, c, fixed, B, beta = problem(rng, m=900)
    h = rng.normal(size=(len(A), 3))  # row identity only
    lookup = {tuple(r): i for i, r in enumerate(h)}
    rows_of = lambda hs: np.array([lookup[tuple(r)] for r in hs])
    sigma = np.full(len(A), 0.3)
    y = 1.7 * A @ c + B @ beta + sigma * rng.normal(size=len(A))
    held = rng.random(len(A)) < 0.2
    train, test = accumulate(lambda hs: A[rows_of(hs)], h, y, sigma,
                             background=lambda hs: B[rows_of(hs)], split=held, tile=128)
    assert train.n_obs == (~held).sum() and test.n_obs == held.sum()
    result = refine_normal(train, fixed=fixed)
    dense = refine(y[~held], sigma[~held], A[~held], fixed=fixed, B=B[~held])
    assert np.allclose(result.c, dense.c, rtol=1e-7)
    model = dense.predict(A[held], B[held])
    direct = np.sum(((y[held] - model) / sigma[held]) ** 2) / held.sum()
    assert np.isclose(held_out_chi2(result, test), direct, rtol=1e-7)


def test_select_columns_matches_subset_refinement():
    rng = np.random.default_rng(5)
    A, c, fixed, B, beta = problem(rng)
    y = A @ c + B @ beta
    neq = NormalEquations(A.shape[1], B.shape[1]).add(A, y, np.ones(len(y)), B)
    cols = np.array([0, 3, 5, 6, 9])
    sub = refine_normal(neq.select(cols), fixed=(np.array([0, 1]), c[[0, 3]]))
    dense = refine(y, np.ones(len(y)), A[:, cols], fixed=(np.array([0, 1]), c[[0, 3]]), B=B)
    assert np.allclose(sub.c, dense.c, rtol=1e-7) and np.isclose(sub.chi2, dense.chi2, rtol=1e-6)
