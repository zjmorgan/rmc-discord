"""Structure application: average structure, disordered sites, supercell."""

import itertools
from dataclasses import replace

import numpy as np

from rmc_discord.crystal import tables
from rmc_discord.crystal.cif import read_cif
from rmc_discord.crystal.structure import Crystal
from rmc_discord.gui.models.base_model import BaseModel, site_name
from rmc_discord.model.supercell import Supercell


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
        self.session.set_crystal(crystal)
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

    def build_supercell(self, cell, occupancies, labels, N, groups=None, ions=None, moments=None,
                        seed=0, displacements=True, progress=None):
        """Rebuild the crystal with edited lattice and occupancies, then the supercell."""
        if progress:
            progress("Building crystal.", 20)
        crystal = self.make_crystal(cell, occupancies)
        if progress:
            progress("Building supercell.", 50)
        sc = Supercell(crystal, N, labels, ions=ions or None, moments=moments or None,
                       groups=groups or None, rng=seed)
        if displacements:
            sc.randomize_displacements()
        if progress:
            progress("Supercell built!", 100)
        return crystal, sc

    def kind_rows(self):
        sc = self.session.supercell
        if sc is None:
            return []
        rows = []
        counts = np.bincount(sc.kind, minlength=len(sc.kinds))
        for k, kind in enumerate(sc.kinds):
            site = sc.sites[kind.site]
            rows.append(dict(site=site_name(site), kind=kind.label, symbol=kind.symbol,
                             fraction=kind.fraction, count=int(counts[k]), b=kind.b.real,
                             moment=kind.moment if kind.magnetic else 0.0,
                             group=len(kind.members) if kind.group else 1))
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

    def supercell_atoms(self):
        sc = self.session.supercell
        r, labels, _, _ = sc.atoms()
        return self._style(r, labels, set(labels), np.ones(len(r)))

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
