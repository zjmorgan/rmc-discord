"""Refined orbit coefficients as pair-correlation tables."""

import numpy as np

from rmc_discord.analysis.correlations import Correlations


def to_correlations(orbits, sites, structure, c, esd=None, quantity="occupancy", names=None, normalized=False):
    """
    Pair table of refined orbit coefficients.

    Every pair of the orbits (both orientations, all symmetry images) gets
    the coefficient and uncertainty of its orbit, so the table supports
    the same filtering, shell averaging, 3D display and HDF5/CSV export
    as correlations computed from configurations.

    Parameters
    ----------
    orbits : PairOrbits
    sites : Sites
        Positions the orbits refer to.
    structure : Structure
        Supplies the Cartesian basis.
    c : ndarray, (n_orbits,)
        Orbit coefficients.
    esd : ndarray, (n_orbits,), optional
        Their standard uncertainties.
    quantity : str, optional
        ``"occupancy"``, ``"spin"`` or ``"displacement"``.
    names : dict, optional
        Display name of each site label (e.g. ``{"Fe1": "Fe1/Mn1"}``).
    normalized : bool, optional
        Coefficients are correlation coefficients.

    Returns
    -------
    corr : rmc_discord.analysis.correlations.Correlations

    """

    c = np.asarray(c, float)
    esd = np.zeros_like(c) if esd is None else np.asarray(esd, float)
    names = names or {}
    labels = [str(names.get(l, l)) for l in sites.label]
    unique = list(dict.fromkeys(sites.label))
    asym = np.array([unique.index(l) for l in sites.label])
    i, j = np.asarray(orbits.i), np.asarray(orbits.j)
    r = np.asarray(orbits.d, float) @ structure.A.T
    o = np.asarray(orbits.orbit)
    return Correlations(
        quantity=quantity,
        sites=np.column_stack([i, j]),
        labels=[f"{labels[a]}-{labels[b]}" for a, b in zip(i, j)],
        orbits=np.column_stack([asym[i], asym[j]]),
        dR=np.asarray(orbits.R, int),
        r=r,
        distance=np.linalg.norm(r, axis=1),
        value=c[o],
        error=esd[o],
        count=np.ones(len(o)),
        tensor=None,
        normalized=normalized,
    )
