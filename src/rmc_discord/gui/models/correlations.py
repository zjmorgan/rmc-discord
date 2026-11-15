"""Correlations application: pair correlations of refined configurations."""

from pathlib import Path

import numpy as np

from rmc_discord.analysis import correlations, write_shells
from rmc_discord.gui.models.base_model import BaseModel

NAMES = {"occupancy": "Occupancy", "displacement": "Displacement", "spin": "Spin"}


class CorrelationsModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.states = []  # (name, dict)
        self.raw = {}
        self.shown = None

    def available(self):
        sc = self.session.supercell
        if sc is None:
            return []
        out = []
        if sc.mixed_sites:
            out.append("occupancy")
        if np.any(sc.u != 0):
            out.append("displacement")
        if any(k.magnetic for k in sc.kinds):
            out.append("spin")
        return out

    def add_state(self, filename):
        sc = self.session.supercell
        state = np.load(filename)
        if state["kind"].shape != sc.kind.shape:
            raise ValueError(f"{Path(filename).name} does not match the supercell")
        entry = {k: np.asarray(state[k]) for k in ("kind", "u", "S") if k in state}
        self.states.append((Path(filename).name, entry))

    def remove_state(self, index):
        if 0 <= index < len(self.states):
            del self.states[index]

    def compute(self, quantity, symmetrize, normalize, use_current, progress=None):
        sc = self.session.supercell
        states = ([{}] if use_current else []) + [s for _, s in self.states]
        if not states:
            raise ValueError("no configurations selected")
        if progress:
            progress(f"Computing {quantity} correlations ({len(states)} configurations).", 20)
        try:
            raw = correlations(sc, quantity, states=states, symmetrize=symmetrize, normalize=False)
            shown = correlations(sc, quantity, states=states, symmetrize=symmetrize, normalize=True) \
                if normalize else raw
        except ValueError as e:
            if not symmetrize:
                raise
            raise ValueError(f"cannot symmetrize ({e}); uncheck Symmetrize") from e
        if progress:
            progress("Correlations computed!", 100)
        return quantity, raw, shown, len(states)

    def set_computed(self, quantity, raw, shown):
        self.raw[quantity] = raw
        self.shown = shown
        self.session.set_correlations(quantity, raw)

    # ------------------------------------------------------------------

    def pair_names(self):
        if self.shown is None:
            return []
        names = sorted({"-".join(sorted(label.split("-"))) for label in self.shown.labels})
        return names

    def filtered(self, pair, r_min, r_max):
        c = self.shown
        if c is None:
            return None
        if pair and pair != "All pairs":
            a, b = pair.split("-")
            c = c.pair(a, b)
        c = c.within(r_max, r_min)
        return c.select(c.distance > 1e-8)

    def shells(self, pair, r_min, r_max, tol=1e-3):
        c = self.filtered(pair, r_min, r_max)
        if c is None or len(c) == 0:
            return None
        return c.shells(tol=tol)

    def save(self, filename):
        self.shown.save(filename)

    def write_shells(self, shells, filename):
        write_shells(shells, filename)
