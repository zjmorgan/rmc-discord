"""Refinement application: refine pair correlations against the observations."""

import time
from dataclasses import dataclass

import numpy as np

from rmc_discord.gui.models.base_model import BaseModel, display_names


@dataclass
class Settings:
    """Fit settings collected from the view."""

    background: str = "radial"
    knot_spacing: float = 0.5
    degree: int = 1
    ridge: float = 0.0


class RefinementModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.records = []
        self.running = False
        self.comparison = None

    def defaults(self):
        return Settings()

    def stop(self):
        """Fits run to completion (they take seconds to minutes)."""

    # ------------------------------------------------------------------

    def fit(self, settings, progress=None):
        """Refine the session's pair model against its observations."""
        model, obs = self.session.model, self.session.sample
        if model is None or obs is None:
            raise ValueError("build the model and sample the data first")
        t = time.time()
        if progress:
            progress("Accumulating normal equations.", 5)

        def report(fraction):
            if progress:
                progress(f"Design columns: {100 * fraction:.0f}% of the observations.", int(5 + 85 * fraction))

        fit = model.fit(obs, background=settings.background, spacing=settings.knot_spacing,
                        degree=settings.degree, ridge=settings.ridge, progress=report)
        if progress:
            progress("Tabulating correlations.", 95)
        tables = model.tables(fit, display_names(self.session.crystal))
        record = dict(channels=", ".join(b.name for b in model.blocks), r_max=model.spec.r_max,
                      free=fit.refinement.rank, chi2_train=fit.chi2_train, chi2_test=fit.chi2_test,
                      n_train=fit.n_train, n_test=fit.n_test, seconds=time.time() - t,
                      scale=fit.refinement.scale, ties=fit.ties)
        if progress:
            progress("Refined!", 100)
        return fit, tables, record

    def set_fit(self, fit, tables, record):
        self.records.append(record)
        self.comparison = None
        self.session.set_fit(fit, tables)

    # ------------------------------------------------------------------

    def compare_slice(self, normal, value, thickness):
        """
        Observed, calculated, and normalized residual on the measured bins of
        a slab normal to a projection axis of the data volume (the model is
        evaluated at every bin, not only at the merged observations).

        Returns
        -------
        out : dict or None
            ``obs``, ``calc``, ``residual`` images, bin ``edges`` of the two
            in-plane axes, their indices ``axes``, the ``transform`` (skew,
            aspect) for the true metric, and the axis ``names``; None if no
            point lies in the slab.
        """
        crystal, v, fit = self.session.crystal, self.session.volume, self.session.fit
        if crystal is None or v is None or fit is None:
            return None
        W = v.W
        c = self._slab(normal, value, thickness)
        if c is None:
            return None
        u = c["hkl"] @ np.linalg.inv(W).T
        keep = np.ones(len(u), bool)
        axes = [a for a in range(3) if a != normal]
        index, edges = [], []
        for a in axes:
            x = u[keep, a]
            levels = np.unique(np.round(x, 6))
            step = np.min(np.diff(levels)) if len(levels) > 1 else 1.0
            i = np.round((x - levels[0]) / step).astype(int)
            index.append(i)
            edges.append(levels[0] + step * (np.arange(i.max() + 2) - 0.5))
        shape = (len(edges[0]) - 1, len(edges[1]) - 1)
        flat = np.ravel_multi_index(index, shape)
        count = np.bincount(flat, minlength=np.prod(shape)).astype(float)

        def image(values):
            with np.errstate(invalid="ignore"):
                return (np.bincount(flat, values, minlength=count.size) / np.where(count > 0, count, np.nan)).reshape(shape)

        obs, calc, sigma = c["obs"][keep], c["calc"][keep], c["sigma"][keep]
        q = crystal.B @ W
        qa, qb = q[:, axes[0]], q[:, axes[1]]
        la, lb = np.linalg.norm(qa), np.linalg.norm(qb)
        cos = qa @ qb / (la * lb)
        from rmc_discord.gui.models.data import DataModel

        return dict(obs=image(obs), calc=image(calc), residual=image((obs - calc) / sigma), edges=edges,
                    axes=tuple(axes), transform=(lb * cos / la, lb * np.sqrt(max(0.0, 1 - cos**2)) / la),
                    names=DataModel.axis_names_of(W), n=int(keep.sum()))

    def selected(self, hkl):
        """Mask of positions inside the observation selection (|Q| range, Bragg exclusion)."""
        info = self.session.sample.info if self.session.sample is not None else {}
        structure = self.session.structure
        Q = np.linalg.norm(structure.Q(hkl), axis=1)
        lo, hi = info.get("q_range", (0.0, np.inf))
        keep = (Q >= lo) & (Q <= hi) & (Q > 0)
        radius = info.get("bragg_radius", 0.0)
        if radius > 0:
            from rmc_discord.pairs.selection import bragg_mask

            keep &= ~bragg_mask(hkl, structure, radius)
        return keep

    def _slab(self, normal, value, thickness):
        """Measured bins in the slab with the model evaluated there (cached)."""
        key = (normal, round(value, 9), round(thickness, 9), id(self.session.fit))
        if self.comparison is not None and self.comparison.get("key") == key:
            return self.comparison
        v, model, fit = self.session.volume, self.session.model, self.session.fit
        centers = v.centers[normal]
        sel = np.flatnonzero(np.abs(centers - value) <= max(thickness / 2, v.widths[normal] / 2) + 1e-9)
        if not sel.size:
            sel = np.array([int(np.argmin(np.abs(centers - value)))])
        signal = np.take(v.signal, sel, axis=normal)
        variance = np.take(v.variance, sel, axis=normal)
        ok = np.isfinite(signal) & np.isfinite(variance) & (variance > 0)
        index = np.argwhere(ok)
        if not len(index):
            return None
        index[:, normal] = sel[index[:, normal]]
        hkl = v.hkl(index)
        obs = signal[ok]
        sigma = np.sqrt(variance[ok])
        keep = self.selected(hkl)  # bins the fit could have used: |Q| range, outside Bragg regions
        if not np.any(keep):
            return None
        hkl, obs, sigma, index = hkl[keep], obs[keep], sigma[keep], index[keep]
        calc = self.session.model_volume("model")[tuple(index.T)]
        self.comparison = dict(key=key, hkl=hkl, obs=obs, calc=calc, sigma=sigma)
        return self.comparison
