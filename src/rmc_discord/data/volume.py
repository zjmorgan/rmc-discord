"""Histogrammed reciprocal-space volumes read with Mantid."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def data_root():
    """
    Directory with private experimental data.

    Returns
    -------
    root : pathlib.Path or None
        ``$DIFFUSE_DATA_ROOT`` if set, else None.

    """

    root = os.environ.get("DIFFUSE_DATA_ROOT")
    return Path(root) if root else None


@dataclass
class Volume:
    r"""
    Regular-grid volume of intensity in reciprocal space.

    Attributes
    ----------
    signal : ndarray, (n0, n1, n2)
        Intensity; NaN where not measured.
    variance : ndarray, (n0, n1, n2)
        Squared uncertainty.
    edges : list of ndarray
        Bin edges along each axis in projection coordinates ``u``.
    W : ndarray, (3, 3)
        Projection matrix, :math:`\mathbf{h} = W \mathbf{u}`.
    lattice : tuple or None
        Oriented-lattice parameters stored with the data, if any.

    """

    signal: np.ndarray
    variance: np.ndarray
    edges: list
    W: np.ndarray
    lattice: tuple = None

    @property
    def shape(self):
        return self.signal.shape

    @property
    def centers(self):
        """Bin centers along each axis."""
        return [0.5 * (e[1:] + e[:-1]) for e in self.edges]

    @property
    def widths(self):
        """Bin widths along each axis."""
        return np.array([e[1] - e[0] for e in self.edges])

    def hkl(self, index):
        """
        Miller indices of bin centers.

        Parameters
        ----------
        index : array_like, (..., 3), int
            Bin indices.

        Returns
        -------
        hkl : ndarray, (..., 3)
            Miller indices.

        """

        index = np.asarray(index)
        u = np.stack(
            [self.edges[a][0] + (index[..., a] + 0.5) * self.widths[a] for a in range(3)],
            axis=-1,
        )
        return u @ self.W.T

    def index(self, hkl):
        """
        Bin containing each point.

        Parameters
        ----------
        hkl : array_like, (..., 3)
            Miller indices.

        Returns
        -------
        index : ndarray, (..., 3), int
            Bin indices.
        inside : ndarray of bool, (...)
            True where the point lies inside the volume.

        """

        u = np.asarray(hkl, dtype=float) @ np.linalg.inv(self.W).T
        index = np.stack(
            [np.floor((u[..., a] - self.edges[a][0]) / self.widths[a]) for a in range(3)],
            axis=-1,
        ).astype(int)
        inside = np.all((index >= 0) & (index < np.array(self.shape)), axis=-1)
        return index, inside

    def __truediv__(self, norm):
        """
        Normalized volume, e.g. ``data / norm``.

        The normalization is treated as exact; bins where it is not
        positive become NaN.
        """
        if self.shape != norm.shape or not np.allclose(self.W, norm.W):
            raise ValueError("volumes are on different grids")
        with np.errstate(divide="ignore", invalid="ignore"):
            n = np.where(norm.signal > 0, norm.signal, np.nan)
            signal = self.signal / n
            variance = self.variance / n**2
        return Volume(
            signal=signal,
            variance=variance,
            edges=self.edges,
            W=self.W,
            lattice=self.lattice,
        )

    def __sub__(self, other):
        """Difference of two volumes on the same grid."""
        if self.shape != other.shape or not np.allclose(self.W, other.W):
            raise ValueError("volumes are on different grids")
        return Volume(
            signal=self.signal - other.signal,
            variance=self.variance + other.variance,
            edges=self.edges,
            W=self.W,
            lattice=self.lattice,
        )


def load(filename):
    """
    Read an MDHistoWorkspace NeXus file with ``LoadMD``.

    Parameters
    ----------
    filename : str or path
        File written by ``SaveMD``.

    Returns
    -------
    volume : Volume
        Signal, variance, bin edges, and projection.

    """

    from mantid.simpleapi import DeleteWorkspace, LoadMD

    name = "__rmc_discord_volume"
    ws = LoadMD(str(filename), LoadHistory=False, OutputWorkspace=name)

    signal = ws.getSignalArray().copy()
    variance = ws.getErrorSquaredArray().copy()

    edges = []
    for a in range(3):
        d = ws.getDimension(a)
        edges.append(np.linspace(d.getMinimum(), d.getMaximum(), d.getNBins() + 1))

    W, lattice = np.eye(3), None
    if ws.getNumExperimentInfo() > 0:
        info = ws.getExperimentInfo(0)
        run = info.run()
        if run.hasProperty("W_MATRIX"):
            value = run.getLogData("W_MATRIX").value
            if isinstance(value, str):
                value = [float(v) for v in value.split(",")]
            W = np.array(value, dtype=float).reshape(3, 3)
        sample = info.sample()
        if sample.hasOrientedLattice():
            ol = sample.getOrientedLattice()
            lattice = (ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma())

    DeleteWorkspace(name)

    bad = ~np.isfinite(signal) | ~np.isfinite(variance)
    signal[bad] = np.nan
    variance[bad] = np.nan

    return Volume(signal=signal, variance=variance, edges=edges, W=W, lattice=lattice)


def save(volume, filename):
    """
    Write a volume as an MDHistoWorkspace with ``SaveMD``.

    Parameters
    ----------
    volume : Volume
        Volume to write.
    filename : str or path
        Output NeXus file.

    """

    from mantid.simpleapi import (
        AddSampleLog,
        CreateMDHistoWorkspace,
        DeleteWorkspace,
        SaveMD,
    )

    name = "__rmc_discord_save"
    names = []
    for a in range(3):
        v = volume.W[:, a]
        names.append("[" + ",".join(f"{c:g}" for c in v) + "]")

    extents = []
    for e in volume.edges:
        extents += [e[0], e[-1]]

    CreateMDHistoWorkspace(
        SignalInput=volume.signal.ravel(order="F"),
        ErrorInput=np.sqrt(volume.variance).ravel(order="F"),
        Dimensionality=3,
        Extents=extents,
        NumberOfBins=list(volume.shape),
        Names=names,
        Units="rlu,rlu,rlu",
        OutputWorkspace=name,
    )
    AddSampleLog(
        Workspace=name,
        LogName="W_MATRIX",
        LogText=",".join(f"{v:.12g}" for v in volume.W.ravel()),
        LogType="String",
    )
    SaveMD(name, Filename=str(filename))
    DeleteWorkspace(name)
