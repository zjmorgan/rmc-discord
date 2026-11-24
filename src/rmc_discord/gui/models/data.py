"""Data application: load reduced volumes and take observations from them."""

import numpy as np

from rmc_discord.data import volume
from rmc_discord.pairs.sampling import from_volume
from rmc_discord.gui.models.base_model import BaseModel


class DataModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.data = None
        self.norm = None
        self.subtract = None
        self.display = None
        self.files = {}

    # ------------------------------------------------------------------

    def load(self, data_file, norm_file=None, subtract_file=None, progress=None):
        """
        Load the data volume, an optional normalization (data are then
        counts), and an optional volume subtracted after normalization.
        """
        if progress:
            progress("Loading data.", 20)
        data = volume.load(data_file)
        norm = sub = None
        if norm_file:
            if progress:
                progress("Loading normalization.", 50)
            norm = volume.load(norm_file)
        if subtract_file:
            if progress:
                progress("Loading subtracted volume.", 70)
            sub = volume.load(subtract_file)
        display = data / norm if norm is not None else data
        if sub is not None:
            display = display - sub
        if progress:
            progress("Data loaded!", 100)
        return dict(data=data, norm=norm, subtract=sub, display=display,
                    files=dict(data=data_file, norm=norm_file, subtract=subtract_file))

    def set_loaded(self, loaded):
        self.data, self.norm, self.subtract = loaded["data"], loaded["norm"], loaded["subtract"]
        self.display, self.files = loaded["display"], loaded["files"]

    def info(self):
        v = self.display
        lattice = v.lattice
        return dict(shape=v.shape, W=v.W, lattice=lattice,
                    ranges=[(e[0], e[-1]) for e in v.edges],
                    measured=float(np.mean(np.isfinite(v.signal))))

    def sample(self, q_min, q_max, bragg_radius, rings, f_sys, merge, extra, held_out, seed=0, progress=None):
        """
        Observations from the displayed volume (bins as measured).

        Laue-equivalent bins are merged (``merge`` "mean" or "first") so
        each observation is one orbit; a fraction is held out.
        """
        if progress:
            progress("Masking and merging equivalent bins.", 20)
        obs = from_volume(self.display, self.session.structure, bragg_radius=bragg_radius, q_range=(q_min, q_max),
                          rings=rings, merge=merge, extra=extra, error_floor=f_sys, held_out=held_out, seed=seed)
        if progress:
            progress("Sampled!", 100)
        info = obs.info
        summary = dict(points=len(obs), bins=info["bins"], bragg=info.get("bragg", 0), rings=info.get("rings", 0),
                       held=int(obs.held.sum()), masked=info["masked"])
        return obs, summary

    # ------------------------------------------------------------------
    # display

    def axis_vectors(self):
        """Reciprocal vectors (1/d units) of the projection axes, columns."""
        return self.crystal.B @ self.display.W

    def axis_names(self):
        """Mantid-style projection axis names, e.g. ``[H,H,0]``, ``[-K,K,0]``, ``[0,0,L]``."""
        return self.axis_names_of(self.display.W)

    @staticmethod
    def axis_names_of(W):
        names = []
        for a, col in enumerate(np.asarray(W).T):
            letter = "HKL"[a]
            parts = []
            for c in col:
                if np.isclose(c, 0):
                    parts.append("0")
                elif np.isclose(c, 1):
                    parts.append(letter)
                elif np.isclose(c, -1):
                    parts.append("-" + letter)
                else:
                    parts.append(f"{c:g}{letter}")
            names.append("[" + ",".join(parts) + "]")
        return names

    def slice(self, normal, value, thickness):
        """
        Mean of the display volume over a slab normal to a projection axis.

        Returns
        -------
        image : ndarray, (n_a, n_b)
        edges_a, edges_b : ndarray
        axes : (int, int)
        transform : (float, float)
            Skew ``ub -> ua`` shift factor and the aspect ratio giving the
            true reciprocal metric.
        """
        v = self.display
        axes = [a for a in range(3) if a != normal]
        centers = v.centers[normal]
        sel = np.abs(centers - value) <= max(thickness / 2, v.widths[normal] / 2)
        if not np.any(sel):
            sel = np.abs(centers - value) == np.min(np.abs(centers - value))
        signal = np.moveaxis(v.signal, normal, 0)[sel]
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            image = np.nanmean(signal, axis=0) if signal.shape[0] > 1 else signal[0]
        q = self.axis_vectors()
        qa, qb = q[:, axes[0]], q[:, axes[1]]
        la, lb = np.linalg.norm(qa), np.linalg.norm(qb)
        cos = qa @ qb / (la * lb)
        sin = np.sqrt(max(0.0, 1 - cos**2))
        return image, v.edges[axes[0]], v.edges[axes[1]], tuple(axes), (lb * cos / la, lb * sin / la)

    def planes(self, positions):
        """
        Three planes of the display volume through the bins nearest
        ``positions`` (one per projection axis), for the 3-D view.

        Returns
        -------
        planes : list of (ndarray, ndarray)
            Cartesian corner points ``(n_b + 1, n_c + 1, 3)`` (1/d units) and
            the plane of the signal ``(n_b, n_c)``.
        """
        v = self.display
        q = self.axis_vectors()
        planes = []
        for a, value in enumerate(positions):
            i = int(np.argmin(np.abs(v.centers[a] - value)))
            b, c = [k for k in range(3) if k != a]
            u = np.zeros((len(v.edges[b]), len(v.edges[c]), 3))
            u[..., a] = v.centers[a][i]
            u[..., b] = v.edges[b][:, None]
            u[..., c] = v.edges[c][None, :]
            planes.append((u @ q.T, np.take(v.signal, i, axis=a)))
        return planes

    def sample_points_in_slice(self, normal, value, thickness):
        s = self.session.sample
        if s is None:
            return None
        return self._in_slice(s.h, normal, value, thickness)

    def masked_points_in_slice(self, normal, value, thickness):
        """Commensurate points removed by the Bragg and powder-ring masks, in the slab."""
        masked = getattr(self, "masked", None)
        if masked is None:
            return None
        return {k: self._in_slice(hkl, normal, value, thickness) for k, hkl in masked.items()}

    def _in_slice(self, hkl, normal, value, thickness):
        u = hkl @ np.linalg.inv(self.display.W).T
        keep = np.abs(u[:, normal] - value) <= thickness / 2
        axes = [a for a in range(3) if a != normal]
        return u[keep][:, axes]

    def sample_cloud(self, max_points=300_000, seed=0):
        """Cartesian Q and log intensity of (up to) ``max_points`` sample points."""
        s = self.session.sample
        if s is None:
            return None
        idx = np.arange(len(s))
        if len(s) > max_points:
            idx = np.sort(np.random.default_rng(seed).choice(len(s), max_points, replace=False))
        y = s.y[idx]
        pos = y[y > 0]
        floor = np.percentile(pos, 1) if pos.size else 1.0
        return 2 * np.pi * s.h[idx] @ self.crystal.B.T, np.log10(np.maximum(y, floor))
