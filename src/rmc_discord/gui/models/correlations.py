"""Correlations application: refined pair correlations of each channel."""

import numpy as np

from rmc_discord.analysis import write_shells
from rmc_discord.gui.models.base_model import BaseModel

NAMES = {
    "occupancy": "Occupancy ⟨δn δn⟩",
    "size effect": "Size Effect (along the bond, Å)",
    "displacement": "Displacement (tr V, Å²)",
    "spin": "Spin ⟨S·S⟩",
}


class CorrelationsModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.shown = None
        self.channel = None
        self.normalized = False
        self.scale = None

    def available(self):
        return list(self.session.correlations)

    def select(self, channel, normalize=False):
        """Table of one channel; occupancies optionally as correlation coefficients."""
        table = self.session.correlations.get(channel)
        self.channel = channel
        self.normalized = False
        if table is not None and normalize and channel == "occupancy":
            block = next(b for b in self.session.model.blocks if b.name == channel)
            c = np.asarray(block.sites.occupancy, float)
            v = c * (1 - c)
            scale = np.sqrt(v[table.sites[:, 0]] * v[table.sites[:, 1]])
            table = table.select(np.arange(len(table)))
            with np.errstate(invalid="ignore", divide="ignore"):
                table.value = table.value / scale
                table.error = table.error / scale
            table.normalized = self.normalized = True
        elif table is not None and self.scale and self.difference():
            table = table.select(np.arange(len(table)))
            table.value = table.value / self.scale
            table.error = table.error / self.scale
        self.shown = table
        return table

    def difference(self):
        """Spin correlations refined from difference data (known only times the scale)."""
        model = self.session.model
        return self.channel == "spin" and model is not None and bool(getattr(model.spec, "difference", False))

    def info(self):
        t = self.shown
        if t is None:
            return ""
        return f"{len(t):,} pair vectors on {len(set(t.labels))} site pairs to {t.distance.max():.2f} Å."

    # ------------------------------------------------------------------

    def pair_names(self):
        if self.shown is None:
            return []
        return sorted({"-".join(sorted(label.split("-"))) for label in self.shown.labels})

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

    def plane(self, pair, hkl, d, tol, r_max):
        """
        Pair vectors on the plane ``(hkl) . [uvw] = d`` within ``tol``.

        Returns in-plane Cartesian coordinates (Angstrom) along ``e1``, the
        shortest lattice direction in the plane, and ``e2 = n x e1``, with
        coincident vectors of different site pairs averaged, or None.
        """
        c = self.filtered(pair, 0.0, r_max)
        hkl = np.asarray(hkl, float)
        if c is None or not np.any(hkl):
            return None
        A, B = self.session.structure.A, self.session.structure.B
        n = B @ hkl
        n = n / np.linalg.norm(n)
        on = np.abs(c.r @ B @ hkl - d) <= tol
        directions = in_plane_directions(hkl, A)
        e1 = A @ directions[0]
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        r, value, error, count = c.r[on], c.value[on], c.error[on], np.maximum(c.count[on], 1)
        if len(r):
            _, first, inverse = np.unique(np.round(r, 4), axis=0, return_index=True, return_inverse=True)
            inverse = inverse.ravel()
            w = np.bincount(inverse, count)
            value = np.bincount(inverse, count * value) / w
            error = np.sqrt(np.bincount(inverse, (count * error) ** 2)) / w
            r = r[first]
        return {"x": r @ e1, "y": r @ e2, "value": value, "error": error, "directions": directions,
                "axes": np.array([[A @ u @ e1, A @ u @ e2] for u in directions]), "hkl": hkl, "d": d}

    def save(self, filename):
        self.shown.save(filename)

    def write_shells(self, shells, filename):
        write_shells(shells, filename)


def in_plane_directions(hkl, A, n=6):
    """The two shortest independent lattice directions [uvw] with ``hkl . uvw = 0``."""
    g = np.arange(-n, n + 1)
    u = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    u = u[(u @ np.asarray(hkl) == 0) & np.any(u != 0, axis=1)]
    divisor = np.gcd.reduce(np.abs(u), axis=1)
    u = np.unique(u[divisor == 1], axis=0)
    length = np.linalg.norm(u @ A.T, axis=1)
    order = np.lexsort((-u[:, 2], -u[:, 1], -u[:, 0], np.round(length, 6)))
    u = u[order]
    first = u[0]
    for v in u[1:]:
        if np.linalg.norm(np.cross(A @ first, A @ v)) > 1e-6 * np.linalg.norm(A @ first) * np.linalg.norm(A @ v):
            return np.array([first, v])
    raise ValueError("no in-plane lattice directions")
