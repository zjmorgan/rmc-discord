"""Structure application: average structure, disordered sites, pair-correlation model."""

import itertools
from dataclasses import replace

import numpy as np

from rmc_discord.crystal import tables
from rmc_discord.crystal.cif import read_cif
from rmc_discord.crystal.structure import Crystal
from rmc_discord.gui.models.base_model import BaseModel, site_name
from rmc_discord.pairs import Structure
from rmc_discord.pairs.workflow import ModelSpec, PairModel


def parse_mapping(text, many=False, number=False):
    """
    Parse ``"A: x; B: y"`` into a dict.

    With ``many`` the values are comma-separated lists; with ``number``
    they are floats.
    """
    out = {}
    for item in text.replace("\n", ";").split(";"):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"expected 'key: value' in {item.strip()!r}")
        key, value = (s.strip() for s in item.split(":", 1))
        if many:
            out[key] = [v.strip() for v in value.split(",") if v.strip()]
        elif number:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def format_mapping(mapping):
    return "; ".join(
        f"{k}: {', '.join(v) if isinstance(v, (list, tuple)) else v}" for k, v in mapping.items()
    )


class StructureModel(BaseModel):
    def __init__(self, session):
        super().__init__(session)
        self.cif = None
        self.filename = None

    # ------------------------------------------------------------------

    def load_cif(self, filename):
        self.cif = read_cif(filename)
        self.filename = str(filename)
        crystal = self.make_crystal(self.cif.cell, {})
        self.session.set_crystal(crystal, Structure.from_crystal(crystal))
        return crystal

    def make_crystal(self, cell, occupancies):
        sites = [replace(s, occupancy=float(occupancies.get(s.label, s.occupancy))) for s in self.cif.sites]
        return Crystal(tuple(cell), self.cif.space_group, sites, operators=self.cif.operators, number=self.cif.number)

    def site_rows(self):
        """Asymmetric sites: label, type, occupancy, x, y, z, U_iso, multiplicity."""
        crystal = self.crystal
        if crystal is None:
            return []
        mult = np.bincount(crystal.atom_index, minlength=len(crystal.asymmetric))
        rows = []
        for k, s in enumerate(crystal.asymmetric):
            U = s.U_iso if s.U is None else float(np.trace(crystal._U_cartesian(s)) / 3)
            rows.append(dict(label=s.label, symbol=s.symbol, occupancy=s.occupancy,
                             x=s.xyz[0], y=s.xyz[1], z=s.xyz[2], U=U, multiplicity=int(mult[k])))
        return rows

    @staticmethod
    def magnetic_positions(crystal, ions, moments):
        """
        Magnetic positions from ``{label: ion}`` and ``{ion or label: moment}``.

        Co-located CIF labels form one position, named by its first label,
        with the occupancy-weighted moment of its magnetic occupants
        (non-magnetic occupants and vacancies carry none).

        Returns
        -------
        labels : list of str
        mixtures : dict
            ``{label: [(occupancy, ion or None), ...]}``.
        by_ion : dict
            ``{ion: moment}``.
        """
        unknown = set(ions) - {s.label for s in crystal.asymmetric}
        if unknown:
            raise ValueError(f"unknown labels in magnetic ions: {sorted(unknown)}")
        by_ion = {}
        for label, ion in ions.items():
            moment = moments.get(label, moments.get(ion))
            if moment is None:
                raise ValueError(f"no moment for {label} ({ion})")
            by_ion[ion] = float(moment)
        groups = []
        for s in crystal.asymmetric:
            for g in groups:
                d = (g[0].xyz - s.xyz + 0.5) % 1 - 0.5
                if np.all(np.abs(d) < 1e-4):
                    g.append(s)
                    break
            else:
                groups.append([s])
        labels, mixtures = [], {}
        for g in groups:
            if not any(s.label in ions for s in g):
                continue
            labels.append(g[0].label)
            mixtures[g[0].label] = [(float(s.occupancy), ions.get(s.label)) for s in g]
        return labels, mixtures, by_ion

    def build_model(self, cell, occupancies, labels, channels, magnetic, ions, moments, r_max, cluster_distance,
                    axis=None, difference=False, progress=None):
        """
        Crystal with the edited lattice and occupancies, and its pair model.

        Parameters
        ----------
        labels : list of str
            Sites with occupational/displacive disorder.
        channels : dict
            ``occupancy``, ``size_effect``, ``displacement`` booleans.
        magnetic : {None, "isotropic", "tensor"}
        ions, moments : dict
            Parsed magnetic ions and moments.
        """
        if progress:
            progress("Building crystal.", 20)
        crystal = self.make_crystal(cell, occupancies)
        structure = Structure.from_crystal(crystal)
        mlabels, mixtures, by_ion = ([], {}, {})
        if magnetic:
            if not ions:
                raise ValueError("give the magnetic ions (label: ion) for a magnetic model")
            mlabels, mixtures, by_ion = self.magnetic_positions(crystal, ions, moments)
        spec = ModelSpec(active=list(labels), r_max=float(r_max), cluster_distance=float(cluster_distance),
                         occupancy=channels.get("occupancy", False), size_effect=channels.get("size_effect", False),
                         displacement=channels.get("displacement", False), axis=axis, magnetic=magnetic,
                         magnetic_sites=mlabels, ions=mixtures, moments=by_ion, difference=difference)
        if progress:
            progress("Finding pair orbits and constraints.", 50)
        model = PairModel(structure, spec)
        if progress:
            progress("Model built!", 100)
        return crystal, structure, model

    def model_rows(self):
        model = self.session.model
        if model is None:
            return []
        rows = []
        for b in model.blocks:
            sites = ", ".join(dict.fromkeys(b.sites.label))
            orbits = len(b.columns) if b.kind == "orbit" else b.orbits.n_orbits
            rows.append(dict(channel=b.name, sites=sites, orbits=orbits, coefficients=b.n,
                             free=b.parameterization.n_free))
        return rows

    # ------------------------------------------------------------------
    # display

    def _symbols(self):
        return {s.label: s.symbol for s in self.crystal.asymmetric}

    def unit_cell_atoms(self, active=()):
        """
        Atoms of the unit cell with boundary copies.

        Returns positions (A), RGB colors, radii, opacity mask (active
        labels opaque), and labels.
        """
        crystal = self.crystal
        xyz = crystal.atom_xyz
        labels = np.array([crystal.asymmetric[i].label for i in crystal.atom_index], dtype=object)
        occ = np.array([crystal.asymmetric[i].occupancy for i in crystal.atom_index])
        all_x, all_l, all_o = [xyz], [labels], [occ]
        for shift in itertools.product([0, 1], repeat=3):
            shift = np.array(shift)
            if not shift.any():
                continue
            keep = np.all((xyz < 1e-3) | (shift == 0), axis=1)
            if np.any(keep):
                all_x.append(xyz[keep] + shift)
                all_l.append(labels[keep])
                all_o.append(occ[keep])
        xyz, labels, occ = np.concatenate(all_x), np.concatenate(all_l), np.concatenate(all_o)
        return self._style(crystal.cartesian(xyz), labels, set(active), occ)

    def _style(self, r, labels, active, occupancy):
        symbols = self._symbols()
        colors, radii = {}, {}
        for label in set(labels):
            symbol = symbols.get(label, label)
            try:
                colors[label] = tables.color(symbol)
            except (KeyError, ValueError):
                colors[label] = (0.6, 0.6, 0.6)
            try:
                radii[label] = tables.radius(symbol)
            except (KeyError, ValueError):
                radii[label] = 1.0
        rgb = np.array([colors[l] for l in labels]).reshape(-1, 3)
        radius = 0.35 * np.array([radii[l] for l in labels])
        opaque = np.array([l in active for l in labels], dtype=bool)
        return dict(r=r, rgb=(255 * rgb).astype(np.uint8), radius=radius, active=opaque, labels=labels,
                    occupancy=occupancy)

    def cell_edges(self, N=(1, 1, 1)):
        """Line segments of the (super)cell outline, (12, 2, 3)."""
        corners = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)
        edges = []
        for i, j in itertools.combinations(range(8), 2):
            if np.sum(np.abs(corners[i] - corners[j])) == 1:
                edges.append((corners[i] * N, corners[j] * N))
        return self.crystal.cartesian(np.array(edges))
