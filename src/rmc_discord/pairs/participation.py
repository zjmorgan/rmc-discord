"""Which atoms participate in the correlated disorder?

Candidate units (sets of site labels, typically one split-site cluster
type each) are switched on and off. A unit that does not participate is
uncorrelated with every other position: all its orbit coefficients
beyond the exact within-cluster values are zero. Every combination of
units is refined on training positions and scored on held-out ones, so
models with more parameters gain nothing unless they predict unseen
intensities better.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from rmc_discord.pairs.constraints import (
    Parameterization,
    parameterize,
    stack_parameterizations,
)
from rmc_discord.pairs.refine import r_factors, refine


@dataclass
class ParticipationModel:
    """
    One combination of participating units.

    Attributes
    ----------
    units : tuple of str
        Participating unit names.
    n_params : int
        Refined parameters (correlations, scale, background).
    chi2_train, chi2_test : float
        Reduced chi-squared on training and held-out positions.
    wR_test : float
        Weighted R factor on held-out positions.
    bic : float
        Bayesian information criterion on the training positions.
    refinement : Refinement

    """

    units: tuple
    n_params: int
    chi2_train: float
    chi2_test: float
    wR_test: float
    bic: float
    refinement: object


def participation_scan(
    y,
    sigma,
    A,
    orbits,
    sites,
    units,
    fixed,
    rows,
    train,
    B=None,
    size=None,
    mechanisms=None,
    anchor=None,
):
    """
    Refine every combination of participating units.

    Parameters
    ----------
    y, sigma : ndarray, (m,)
        Observed intensities and uncertainties.
    A : ndarray, (m, n_orbits)
        Design columns.
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating candidate positions.
    units : dict
        ``{name: [site labels]}`` candidate units.
    fixed : tuple of ndarray
        Coefficients fixed by the average structure.
    rows : ndarray, (r, n_orbits)
        Exact constraints (sum rules, homometric ties).
    train : ndarray of bool, (m,)
        Training positions; the rest are held out.
    B : ndarray, (m, nb), optional
        Background basis.
    size : dict, optional
        Shorthand for one mechanism named ``"size effect"``.
    mechanisms : list of dict, optional
        Additional coefficient blocks, each ``{"name": label, "A":
        design columns, "orbit": orbit of each column, "rows": their
        constraint rows}`` (e.g. size effect, displacement
        correlations). Every unit combination is refined with every
        subset of mechanisms; their coefficients follow the same
        participation switches as the occupational ones.
    anchor : tuple, optional
        ``(A_fixed, values)``: columns with fixed coefficients included in
        every model, e.g. the Bragg intensity of the average structure
        with coefficient one, which anchors the common scale.

    Returns
    -------
    models : list of ParticipationModel
        Sorted by held-out reduced chi-squared.

    """

    label_unit = {}
    for name, labels in units.items():
        for label in labels:
            label_unit[label] = name
    reps = orbits.representatives()
    unit_i = np.array([label_unit.get(sites.label[orbits.i[p]]) for p in reps])
    unit_j = np.array([label_unit.get(sites.label[orbits.j[p]]) for p in reps])
    is_fixed = np.zeros(orbits.n_orbits, bool)
    is_fixed[fixed[0]] = True

    test = ~train
    B = np.zeros((len(y), 0)) if B is None else B
    models = []
    names = list(units)
    mechanisms = list(mechanisms or [])
    if size is not None:
        mechanisms.insert(0, dict(size, name="size effect"))
    subsets = [
        combo
        for k in range(len(mechanisms) + 1)
        for combo in itertools.combinations(range(len(mechanisms)), k)
    ]
    for k in range(len(names) + 1):
        for chosen in itertools.combinations(names, k):
            on = np.isin(unit_i, chosen) & np.isin(unit_j, chosen)
            off = np.flatnonzero(~on & ~is_fixed)
            zero = np.zeros((len(off), orbits.n_orbits))
            zero[np.arange(len(off)), off] = 1
            p_occ = parameterize(
                orbits.n_orbits, fixed, np.vstack([rows, zero])
            )
            for subset in subsets:
                if subset and not chosen:
                    continue
                parts, blocks, label = [p_occ], [A], chosen
                for m in subset:
                    mech = mechanisms[m]
                    n_m = mech["A"].shape[1]
                    off_m = np.flatnonzero(~on[mech["orbit"]])
                    zero_m = np.zeros((len(off_m), n_m))
                    zero_m[np.arange(len(off_m)), off_m] = 1
                    parts.append(
                        parameterize(
                            n_m, None, np.vstack([mech["rows"], zero_m])
                        )
                    )
                    blocks.append(mech["A"])
                    label = label + (mech["name"],)
                if anchor is not None:
                    A_fixed, values = anchor
                    A_fixed = np.asarray(A_fixed).reshape(len(y), -1)
                    parts.insert(
                        0,
                        Parameterization(
                            np.atleast_1d(values).astype(float),
                            np.zeros((A_fixed.shape[1], 0)),
                            0.0,
                        ),
                    )
                    blocks.insert(0, A_fixed)
                p = stack_parameterizations(*parts)
                A_all = np.hstack(blocks)
                r = refine(
                    y[train],
                    sigma[train],
                    A_all[train],
                    B=B[train],
                    parameterization=p,
                )
                model = r.predict(A_all[test], B[test])
                chi2_test = np.sum(((y[test] - model) / sigma[test]) ** 2)
                _, wR = r_factors(y[test], sigma[test], model)
                n = train.sum()
                models.append(
                    ParticipationModel(
                        label,
                        r.n_params,
                        r.reduced_chi2,
                        chi2_test / test.sum(),
                        wR,
                        r.chi2 + r.n_params * np.log(n),
                        r,
                    )
                )
    return sorted(models, key=lambda m: m.chi2_test)
