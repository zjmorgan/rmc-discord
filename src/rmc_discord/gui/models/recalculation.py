"""Recalculation application: the refined model on a data plane."""

import warnings

import numpy as np

from rmc_discord.gui.models.base_model import BaseModel
from rmc_discord.gui.models.data import DataModel


class RecalculationModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.result = None

    def channels(self):
        return list(self.session.correlations)

    def _plane(self, normal, value):
        v = self.session.volume
        axes = [a for a in range(3) if a != normal]
        centers = v.centers
        k = int(np.argmin(np.abs(centers[normal] - value)))
        self._index = k
        signal = np.take(v.signal, k, axis=normal)
        variance = np.take(v.variance, k, axis=normal)
        ua, ub = np.meshgrid(centers[axes[0]], centers[axes[1]], indexing="ij")
        u = np.zeros(ua.shape + (3,))
        u[..., axes[0]], u[..., axes[1]], u[..., normal] = ua, ub, centers[normal][k]
        hkl = u.reshape(-1, 3) @ v.W.T
        return signal, variance, hkl, axes, float(centers[normal][k])

    def recalculate(self, component, normal, value, q_min, q_max, bragg, progress=None):
        """
        The refined model (or one channel, or the background) on a plane of
        the data volume, with the scale and background of the fit.
        """
        s = self.session
        model, fit, v = s.model, s.fit, s.volume
        if model is None or fit is None:
            raise ValueError("refine the correlations first")
        signal, variance, hkl, axes, at = self._plane(normal, value)
        y = signal.ravel()
        Qn = np.linalg.norm(self.crystal.Q(hkl), axis=1)
        ok = np.isfinite(y) & (Qn >= q_min) & (Qn < q_max) & (Qn > 0)
        if bragg > 0:
            ok &= ~np.all(np.abs(hkl - np.round(hkl)) < bragg, axis=1)
        if not np.any(ok):
            raise ValueError("no data points in the plane after masking")
        if progress:
            progress("Evaluating the model on the volume (NUFFT, once per fit).", 20)

        def plane(component):
            return np.take(s.model_volume(component), self._index, axis=normal).ravel()[ok]

        full = plane("model")
        has_background = fit.background is not None and len(fit.refinement.background)
        background = plane("background") if has_background else np.zeros(ok.sum())
        ref = fit.refinement
        if component == "model":
            part, title = full, "model"
        elif component == "background":
            part, title = background, "background"
        else:
            part, title = plane(component), component
        r_all = float(np.corrcoef(full, y[ok])[0, 1])
        r_diffuse = float(np.corrcoef(full - background, y[ok] - background)[0, 1])
        sigma = np.sqrt(variance.ravel()[ok])
        good = np.isfinite(sigma) & (sigma > 0)
        chi2 = float(np.mean(((y[ok] - full)[good] / sigma[good]) ** 2)) if np.any(good) else np.nan
        image = np.full(len(hkl), np.nan)
        image[ok] = part
        data = np.where(ok, y, np.nan)
        shape = signal.shape
        names = DataModel.axis_names_of(v.W)
        if progress:
            progress("Recalculated!", 100)
        return dict(data=data.reshape(shape), model=image.reshape(shape), edges=(v.edges[axes[0]], v.edges[axes[1]]),
                    axes=axes, normal=normal, value=at, names=names, transform=self._transform(axes),
                    r=r_all, r_diffuse=r_diffuse, chi2=chi2, scale=float(ref.scale), n=int(ok.sum()),
                    title=title, Q=self.crystal.Q(hkl).reshape(shape + (3,)))

    def _transform(self, axes):
        q = self.crystal.B @ self.session.volume.W
        qa, qb = q[:, axes[0]], q[:, axes[1]]
        la, lb = np.linalg.norm(qa), np.linalg.norm(qb)
        cos = qa @ qb / (la * lb)
        return lb * cos / la, lb * np.sqrt(max(0.0, 1 - cos**2)) / la
