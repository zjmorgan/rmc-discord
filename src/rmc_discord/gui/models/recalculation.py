"""Recalculation application: diffuse intensity from correlations, compared with a data plane."""

import warnings

import numpy as np

from rmc_discord.analysis import max_range, recalculate
from rmc_discord.gui.models.base_model import BaseModel
from rmc_discord.gui.models.data import DataModel


class RecalculationModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.result = None

    def quantities(self):
        return list(self.session.correlations)

    def max_range(self):
        sc = self.session.supercell
        return None if sc is None else max_range(sc)

    def _plane(self, normal, value):
        v = self.session.volume
        axes = [a for a in range(3) if a != normal]
        centers = v.centers
        k = int(np.argmin(np.abs(centers[normal] - value)))
        signal = np.take(v.signal, k, axis=normal)
        ua, ub = np.meshgrid(centers[axes[0]], centers[axes[1]], indexing="ij")
        u = np.zeros(ua.shape + (3,))
        u[..., axes[0]], u[..., axes[1]], u[..., normal] = ua, ub, centers[normal][k]
        hkl = u.reshape(-1, 3) @ v.W.T
        return signal, hkl, axes, float(centers[normal][k])

    def recalculate(self, quantity, normal, value, r_max, q_min, q_max, bragg, degree, radiation,
                    progress=None):
        sc, corr = self.session.supercell, self.session.correlations[quantity]
        signal, hkl, axes, at = self._plane(normal, value)
        y = signal.ravel()
        Qn = np.linalg.norm(self.crystal.Q(hkl), axis=1)
        ok = np.isfinite(y) & (Qn >= q_min) & (Qn < q_max)
        if bragg > 0:
            ok &= ~np.all(np.abs(hkl - np.round(hkl)) < bragg, axis=1)
        if not np.any(ok):
            raise ValueError("no data points in the plane after masking")
        if progress:
            progress(f"Recalculating {ok.sum():,} points.", 20)
        I = np.full(len(hkl), np.nan)
        I[ok] = recalculate(sc, corr, hkl[ok], r_max=r_max, radiation=radiation)
        if progress:
            progress("Fitting scale and background.", 80)
        A = np.column_stack([I[ok]] + [Qn[ok] ** p for p in range(degree + 1)])
        coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        model = np.full(len(hkl), np.nan)
        model[ok] = A @ coef
        background = A[:, 1:] @ coef[1:]
        r_all = float(np.corrcoef(model[ok], y[ok])[0, 1])
        r_diffuse = float(np.corrcoef(I[ok], y[ok] - background)[0, 1])
        data = np.where(ok, y, np.nan)
        shape = signal.shape
        v = self.session.volume
        names = DataModel.axis_names_of(v.W)
        if progress:
            progress("Recalculated!", 100)
        return dict(data=data.reshape(shape), model=model.reshape(shape), edges=(v.edges[axes[0]], v.edges[axes[1]]),
                    axes=axes, normal=normal, value=at, names=names, transform=self._transform(axes),
                    r=r_all, r_diffuse=r_diffuse, scale=float(coef[0]), n=int(ok.sum()),
                    Q=self.crystal.Q(hkl).reshape(shape + (3,)))

    def _transform(self, axes):
        q = self.crystal.B @ self.session.volume.W
        qa, qb = q[:, axes[0]], q[:, axes[1]]
        la, lb = np.linalg.norm(qa), np.linalg.norm(qb)
        cos = qa @ qb / (la * lb)
        return lb * cos / la, lb * np.sqrt(max(0.0, 1 - cos**2)) / la
