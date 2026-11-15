"""Refinement application: build a refinement and anneal it with held-out validation."""

import threading
from dataclasses import dataclass, field

import numpy as np

from rmc_discord.engine.anneal import Schedule, anneal, split_sample
from rmc_discord.engine.refinement import MOVES, Refinement
from rmc_discord.gui.models.base_model import BaseModel, site_name


@dataclass
class Settings:
    """Refinement and validation settings collected from the view."""

    nuclear: bool = True
    magnetic: bool = False
    radiation: str = "neutron"
    offset: str = None
    weights: dict = field(default_factory=lambda: {"swap": 1.0})
    displacive: list = field(default_factory=list)
    penalty: float = 10.0
    step: float = 0.5
    cone: float = 60.0
    background: str = "radial"
    knot_spacing: float = 0.5
    symmetrize: list = field(default_factory=list)
    held_out: float = 0.1
    n_held: int = None
    planes: tuple = None  # (axis, positions, half width)
    ratio: float = None   # off-plane : on-plane points in the pool
    seed: int = 1


class RefinementModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.records = []
        self.snapshot = None
        self.stop_event = threading.Event()
        self.baseline = {}
        self.running = False

    # ------------------------------------------------------------------

    def defaults(self):
        """Suggested settings for the session's supercell."""
        sc = self.session.supercell
        st = Settings()
        if sc is None:
            return st
        magnetic = any(k.magnetic for k in sc.kinds)
        st.magnetic = magnetic
        st.weights = {}
        if sc.mixed_sites:
            st.weights["swap"] = 1.0
        if magnetic:
            st.weights["spin"] = 1.0
            st.offset = "paramagnetic"
        return st

    def site_labels(self):
        sc = self.session.supercell
        return [] if sc is None else [site_name(s) for s in sc.sites]

    def strata(self, st):
        sample = self.session.sample
        if st.planes is None:
            return None
        axis, positions, width = st.planes
        x = np.abs(sample.hkl[:, axis])
        on = np.min(np.abs(x[:, None] - np.abs(np.asarray(positions))[None]), axis=1) <= width
        return {"on": on, "off": ~on}

    def initialize(self, st, progress=None):
        sample, sc = self.session.sample, self.session.supercell
        if progress:
            progress("Splitting held-out points.", 10)
        strata = self.strata(st)
        pool, held = split_sample(sample, held_out=st.held_out, strata=strata, n_held=st.n_held, seed=0)
        if strata is not None and st.ratio is not None:
            rng = np.random.default_rng(0)
            on_mask = strata["on"]
            on, off = pool[on_mask[pool]], rng.permutation(pool[~on_mask[pool]])
            pool = np.sort(np.concatenate([on, off[: int(st.ratio * len(on))]]))
        if progress:
            progress("Building refinement.", 30)
        ref = Refinement(
            sc, sample.select(pool), nuclear=st.nuclear, magnetic=st.magnetic, radiation=st.radiation,
            offset=st.offset, displacive=st.displacive, penalty=st.penalty, step=st.step, cone=st.cone,
            weights={k: st.weights.get(k, 0.0) for k in MOVES}, background_basis=st.background,
            knot_spacing=st.knot_spacing, symmetrize=st.symmetrize or None, seed=st.seed + 1,
        )
        if progress:
            progress("Scoring the background-only model.", 80)
        baseline = {name: ref.score(h, scale=0.0) for name, h in held.items()}
        if progress:
            progress("Refinement ready!", 100)
        info = dict(pool=len(pool), held={k: len(v) for k, v in held.items()}, chi=ref.reduced_chi_sq,
                    baseline=baseline, atoms=len(sc), moves={k: v for k, v in zip(MOVES, np.diff(ref.move_cdf, prepend=0))
                                                              if v > 0})
        return ref, held, info

    def set_initialized(self, ref, held, info):
        self.records = []
        self.snapshot = None
        self.baseline = info["baseline"]
        self.session.result = None
        self.session.set_refinement(ref, held)

    # ------------------------------------------------------------------

    def run(self, schedule, progress=None):
        ref, held = self.session.refinement, self.session.held
        sc = ref.supercell
        self.stop_event.clear()
        self.running = True
        total = schedule.rounds

        def callback(row, result):
            self.records.append(row)
            self.snapshot = dict(kind=sc.kind.copy(), S=sc.S.copy(), u=sc.u.copy())
            if progress:
                best = result.best["score"] if result.best else np.nan
                progress(f"Round {row['round'] + 1}/{total}: score {row['score']:.4f} (best {best:.4f})",
                         int(100 * (row["round"] + 1) / total))

        try:
            result = anneal(ref, held, schedule, log=print, callback=callback, stop=self.stop_event.is_set)
        finally:
            self.running = False
        self.snapshot = dict(kind=sc.kind.copy(), S=sc.S.copy(), u=sc.u.copy())
        return result

    def stop(self):
        self.stop_event.set()

    def save_configuration(self, filename):
        sc = self.session.supercell
        np.savez(filename, kind=sc.kind, u=sc.u, S=sc.S, N=np.array(sc.N))

    def load_configuration(self, filename):
        sc = self.session.supercell
        state = np.load(filename)
        if "N" in state and tuple(state["N"]) != tuple(sc.N):
            raise ValueError(f"configuration is for N={tuple(state['N'])}, supercell is {tuple(sc.N)}")
        if state["kind"].shape != sc.kind.shape:
            raise ValueError("configuration does not match the supercell")
        sc.kind[:] = state["kind"]
        sc.u[:] = state["u"]
        if "S" in state:
            sc.S[:] = state["S"]
        if self.session.refinement is not None:
            self.session.refinement.recompute(track_drift=False)
        self.snapshot = dict(kind=sc.kind.copy(), S=sc.S.copy(), u=sc.u.copy())
        self.session.configuration_changed.emit()


def schedule_from(values):
    """Schedule from view values (None for blanks)."""
    keys = ["rounds", "sweeps", "cool", "temperature", "batch", "batch_sweeps", "check_every", "adapt",
            "scale_floor"]
    kw = {k: v for k, v in zip(keys, values)
          if v is not None or k in ("temperature", "batch", "adapt", "scale_floor")}
    if "rounds" in kw:
        kw["rounds"] = int(kw["rounds"])
    if kw.get("batch") is not None:
        kw["batch"] = int(kw["batch"])
    if "check_every" in kw:
        kw["check_every"] = int(kw["check_every"])
    return Schedule(**kw)
