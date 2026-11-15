"""
Annealing with minibatches and held-out validation.

The recipe that avoids overfitting a fixed set of points: refine against a
new random minibatch of a training pool every few sweeps, cool
geometrically, score held-out points at the current scale every few rounds,
and keep the configuration that scores best there.
"""

from dataclasses import dataclass, field

import numpy as np

from rmc_discord.engine import kernels


def split_sample(sample, held_out=0.1, strata=None, n_held=None, seed=0):
    """
    Split a sample into a training pool and held-out sets.

    Parameters
    ----------
    sample : QSample
        All points.
    held_out : float, optional
        Fraction of the points (of each stratum) held out.
    strata : dict of str to ndarray of bool, optional
        Named point masks (e.g. on and off diffuse features), each giving
        its own held-out set. One set named ``"held"`` if omitted.
    n_held : int, optional
        Number of held-out points per set, instead of ``held_out``.
    seed : int, optional
        Random seed of the split.

    Returns
    -------
    pool : ndarray of int
        Indices of the training points.
    held : dict of str to QSample
        Held-out points of each stratum.

    """

    rng = np.random.default_rng(seed)
    if strata is None:
        strata = {"held": np.ones(len(sample), dtype=bool)}
    taken = np.zeros(len(sample), dtype=bool)
    held = {}
    for name, mask in strata.items():
        idx = rng.permutation(np.flatnonzero(np.asarray(mask) & ~taken))
        n = int(n_held) if n_held is not None else int(round(held_out * len(idx)))
        pick = np.sort(idx[:n])
        taken[pick] = True
        held[name] = sample.select(pick)
    return np.flatnonzero(~taken), held


@dataclass
class Schedule:
    """
    Annealing schedule.

    Attributes
    ----------
    rounds : int
        Number of rounds.
    sweeps : float
        Sweeps (moves per atom) per round.
    cool : float
        Temperature factor per round.
    temperature : float or None
        Starting temperature; calibrated on a minibatch if None.
    batch : int or None
        Points per minibatch; all pool points if None.
    batch_sweeps : float
        Sweeps between minibatch redraws.
    check_every : int
        Rounds between held-out checks.
    adapt : float or None
        Target acceptance for adapting the displacement step and spin cone
        after each minibatch; no adaptation if None.
    scale_floor : float or None
        If set, the scale is held fixed during each round at
        ``max(free scale, scale_floor * matched scale)`` and refit at each
        check; this keeps a free scale from collapsing to zero, where no move
        changes chi^2. A free scale throughout if None.

    """

    rounds: int = 60
    sweeps: float = 4.0
    cool: float = 0.92
    temperature: float = None
    batch: int = 100_000
    batch_sweeps: float = 2.0
    check_every: int = 5
    adapt: float = None
    scale_floor: float = None


@dataclass
class AnnealResult:
    """
    Outcome of :func:`anneal`.

    Attributes
    ----------
    baseline : dict of str to float
        Held-out reduced chi^2 of the background alone (scale 0).
    records : list of dict
        One row per check: round, temperature, pool chi^2, scale, held-out
        chi^2 of each set, score, and acceptance of each move type.
    best : dict
        Best configuration (``kind``, ``u``, ``S``) and its ``round`` and
        ``score``.

    """

    baseline: dict
    records: list = field(default_factory=list)
    best: dict = None


def _snapshot(sc):
    return {"kind": sc.kind.copy(), "u": sc.u.copy(), "S": sc.S.copy()}


def anneal(ref, held, schedule=None, restore_best=True, save=None, log=print, callback=None, stop=None):
    """
    Anneal a refinement with minibatches and held-out validation.

    Parameters
    ----------
    ref : Refinement
        Refinement built on the training pool (modified in place).
    held : dict of str to QSample
        Held-out point sets (see :func:`split_sample`).
    schedule : Schedule, optional
        Annealing schedule.
    restore_best : bool, optional
        Leave the supercell in the best configuration at the end.
    save : str or Path, optional
        Write the best configuration to this ``.npz`` whenever it improves.
    log : callable or None, optional
        Receives one line of progress per check.
    callback : callable, optional
        Called as ``callback(row, result)`` after each check, with the row
        just added to ``result.records``.
    stop : callable, optional
        Polled after each round; when it returns True the anneal checks the
        current configuration and ends early.

    Returns
    -------
    result : AnnealResult
        Baselines, per-check records, and the best configuration.

    Notes
    -----
    The score of a configuration is the mean over held-out sets of its
    reduced chi^2 divided by the background-only baseline, so below 1 means
    better than no model.

    """

    s = schedule or Schedule()
    sc = ref.supercell
    n_pool = len(ref.full_sample)
    frac = None if s.batch is None or s.batch >= n_pool else s.batch / n_pool

    baseline = {name: ref.score(h, scale=0.0) for name, h in held.items()}
    result = AnnealResult(baseline=baseline)
    if log:
        log("background-only held-out chi2: " + ", ".join(f"{k} {v:.4g}" for k, v in baseline.items()))

    floor = None
    if s.scale_floor is not None:
        floor = s.scale_floor * ref.matched_scale()
        ref.scale = max(ref.scale, floor)
        ref.fit_mode = kernels.FIT_BACKGROUND
        ref.recompute(track_drift=False)

    if s.temperature is None:
        if frac is not None:
            ref.resample(frac)
        ref.calibrate_temperature()
        if frac is not None:
            ref.resample(1.0)
    else:
        ref.temperature = float(s.temperature)
    if log:
        log(f"starting temperature {ref.temperature:.4g}")

    chunk = max(1, int(round(s.batch_sweeps * len(sc))))
    best = np.inf
    for it in range(s.rounds):
        h = ref.run(int(round(s.sweeps * len(sc))), subset=frac, recompute_every=chunk, adapt=s.adapt,
                    restore=False)
        ref.temperature *= s.cool
        stopping = stop is not None and bool(stop())
        if (it + 1) % s.check_every and it != s.rounds - 1 and not stopping:
            continue
        if frac is not None:
            ref.resample(1.0)
        if floor is not None:
            ref.fit_mode = kernels.FIT_FREE
            ref.recompute(track_drift=False)
            free_chi, free_scale = ref.reduced_chi_sq, ref.scale
            ref.scale = max(free_scale, floor)
            ref.fit_mode = kernels.FIT_BACKGROUND
            ref.recompute(track_drift=False)
            score_scale = free_scale
        else:
            free_chi, score_scale = ref.reduced_chi_sq, ref.scale
        chi = {name: ref.score(hs, scale=score_scale) for name, hs in held.items()}
        score = float(np.mean([chi[k] / baseline[k] for k in held]))
        moves = h.array("move")
        acc = {name: h.acceptance(m) for m, name in enumerate(("swap", "spin", "displace", "swap_displace", "exchange"))
               if np.any(moves == m)}
        row = dict(round=it, temperature=ref.temperature, pool=free_chi, scale=score_scale,
                   held=chi, score=score, acceptance=acc)
        improved = score < best
        if improved:
            best = score
            result.best = dict(_snapshot(sc), round=it, score=score)
            if save is not None:
                np.savez(save, **result.best)
        result.records.append(row)
        if log:
            log(f"{it:4d} T {ref.temperature:.3g} pool {free_chi:.4g} scale {score_scale:.3g} | "
                + " ".join(f"{k} {v:.4g}" for k, v in chi.items())
                + f" | score {score:.4f} | acc "
                + "/".join(f"{v:.2f}" for v in acc.values())
                + (" *" if improved else ""))
        if callback is not None:
            callback(row, result)
        if stopping:
            break

    if restore_best and result.best is not None:
        sc.kind[:] = result.best["kind"]
        sc.u[:] = result.best["u"]
        sc.S[:] = result.best["S"]
        if floor is not None:
            ref.fit_mode = kernels.FIT_FREE
        ref.recompute(track_drift=False)
    return result
