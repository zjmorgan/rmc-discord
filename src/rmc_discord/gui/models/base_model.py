"""Base model: lattice axes of the session's crystal for the camera."""

import numpy as np


def site_name(site):
    """Label of a unit-cell site with its position, e.g. ``C2 (0.333, 0.667, 0.187)``."""
    x = ", ".join(f"{v:.3f}" for v in site.xyz)
    return f"{'/'.join(site.labels)} ({x})"


def display_names(crystal):
    """
    Display name of each CIF label: co-located labels joined, e.g.
    ``{"Fe1": "Fe1/Mn1", "Mn1": "Fe1/Mn1"}``.
    """
    if crystal is None:
        return {}
    groups = []
    for s in crystal.asymmetric:
        for g in groups:
            d = (g[0].xyz - s.xyz + 0.5) % 1 - 0.5
            if np.all(np.abs(d) < 1e-4):
                g.append(s)
                break
        else:
            groups.append([s])
    return {s.label: "/".join(t.label for t in g) for g in groups for s in g}


class BaseModel:
    def __init__(self, session):
        self.session = session

    @property
    def crystal(self):
        return self.session.crystal

    def get_lattice_parameters(self):
        if self.crystal is not None:
            return self.crystal.cell

    def get_model_info(self):
        """Space group and a summary of the pair model (channels, range, orbits, coefficients)."""
        if self.crystal is None:
            return None
        group = self.crystal.space_group.getHMSymbol()
        model = self.session.model
        if model is None:
            return group, "", "", "", "", ""
        channels = ", ".join(b.name for b in model.blocks)
        orbits = sum(b.orbits.n_orbits for b in model.blocks if b.kind == "orbit") or model.blocks[0].orbits.n_orbits
        return group, channels, f"{model.spec.r_max:g} Å", orbits, model.n_coefficients, model.n_free

    def orientation_matrix(self):
        """Reciprocal basis in Cartesian coordinates (columns a*, b*, c*)."""
        return self.crystal.B

    def get_transform(self, reciprocal=True):
        """Normalized reciprocal (B) or direct (A) basis vectors as columns."""
        if self.crystal is None:
            return None
        T = self.crystal.B if reciprocal else self.crystal.A
        return T / np.linalg.norm(T, axis=0)

    def get_vector(self, axes_type, ind):
        """Cartesian direction of Miller indices [hkl] or a lattice vector [uvw]."""
        if self.crystal is None:
            return None
        if axes_type == "[hkl]":
            return self.crystal.B @ ind
        return self.crystal.A @ ind

    def _axes(self, i, j):
        if self.crystal is not None:
            B = self.crystal.B
            return B[:, i], B[:, j]

    def bc_star_axes(self):
        return self._axes(0, 1)

    def ca_star_axes(self):
        return self._axes(1, 2)

    def ab_star_axes(self):
        return self._axes(2, 0)

    def bc_axes(self):
        if self.crystal is not None:
            return np.cross(*self.ca_star_axes()), np.cross(*self.ab_star_axes())

    def ca_axes(self):
        if self.crystal is not None:
            return np.cross(*self.ab_star_axes()), np.cross(*self.bc_star_axes())

    def ab_axes(self):
        if self.crystal is not None:
            return np.cross(*self.bc_star_axes()), np.cross(*self.ca_star_axes())

    # ------------------------------------------------------------------
    # camera rotations (as in NeuXtalViz)

    @staticmethod
    def _rotate_vector(v, axis, angle_deg):
        """Rodrigues rotation of ``v`` about ``axis``."""
        v = np.asarray(v, dtype=float)
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n == 0.0:
            return v
        k = axis / n
        t = np.deg2rad(angle_deg)
        return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1.0 - np.cos(t))

    @staticmethod
    def _orthonormal(d, up):
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([0.0, 0.0, 1.0])
        up = np.asarray(up, dtype=float)
        up = up - np.dot(up, d) * d
        if np.linalg.norm(up) == 0.0:
            up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = up - np.dot(up, d) * d
        return d, up / np.linalg.norm(up)

    def _camera(self, position, focal_point, up):
        fp = np.asarray(focal_point, dtype=float)
        d = fp - np.asarray(position, dtype=float)
        dist = np.linalg.norm(d) or 1.0
        d, up = self._orthonormal(d, up)
        return fp, d, up, dist

    def _state(self, fp, d, up, dist):
        d, up = self._orthonormal(d, up)
        return fp - d * dist, fp, up

    def rotate_camera_roll(self, position, focal_point, up, angle_deg):
        """Roll about the viewing axis (positive is counter-clockwise)."""
        fp, d, up, dist = self._camera(position, focal_point, up)
        return self._state(fp, d, self._rotate_vector(up, d, angle_deg), dist)

    def rotate_camera_elevation(self, position, focal_point, up, angle_deg):
        """Tilt about the camera's right axis (positive tilts up)."""
        fp, d, up, dist = self._camera(position, focal_point, up)
        right = np.cross(d, up)
        return self._state(fp, self._rotate_vector(d, right, angle_deg), self._rotate_vector(up, right, angle_deg),
                           dist)

    def rotate_camera_azimuth(self, position, focal_point, up, angle_deg):
        """Rotate about the world z axis (positive turns right)."""
        fp, d, up, dist = self._camera(position, focal_point, up)
        z = np.array([0.0, 0.0, 1.0])
        return self._state(fp, self._rotate_vector(d, z, angle_deg), self._rotate_vector(up, z, angle_deg), dist)

    @staticmethod
    def calculate_camera_parameters(direction, roll):
        """Roll, elevation, and azimuth (degrees) of a viewing direction."""
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([0.0, 0.0, 1.0])
        elevation = np.rad2deg(np.arcsin(np.clip(d[2], -1, 1)))
        azimuth = np.rad2deg(np.arctan2(d[0], d[1]))
        return roll, elevation, azimuth
