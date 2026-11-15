"""Colors and sizes of supercell atoms for the 3D views."""

import numpy as np

from rmc_discord.crystal import tables


def _symbol_map(crystal):
    return {s.label: s.symbol for s in crystal.asymmetric}


def element_style(symbol):
    try:
        rgb = tables.color(symbol)
    except (KeyError, ValueError):
        rgb = (0.6, 0.6, 0.6)
    try:
        r = tables.radius(symbol)
    except (KeyError, ValueError):
        r = 1.0
    return rgb, r


PALETTE = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#9334e6", "#00acc1", "#ff6d01", "#5f6368"]


def _rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def configuration(sc, kind=None, S=None, u=None):
    """
    Occupied supercell atoms (group centers) colored by kind.

    Atoms are drawn with a radius set by the mean site spacing, so they stay
    visible in large supercells; each occupied kind gets a palette color.

    Returns
    -------
    style : dict
        ``r`` positions (A), ``rgb`` uint8 colors, ``radius``, ``kind``
        index, ``S`` spins (or None), ``legend`` [(label, rgb)].
    """
    kind = sc.kind if kind is None else kind
    u = sc.u if u is None else u
    labels = []
    for k in sc.kinds:
        if not k.vacancy and k.label not in labels:
            labels.append(k.label)
    color = {label: _rgb(PALETTE[n % len(PALETTE)]) for n, label in enumerate(labels)}
    rgb = np.array([color.get(k.label, (0.6, 0.6, 0.6)) for k in sc.kinds])
    volume = abs(np.linalg.det(sc.crystal.A))
    spacing = (volume / max(1, sc.n_sites)) ** (1 / 3)
    occupied = np.array([not k.vacancy for k in sc.kinds])[kind]
    idx = np.flatnonzero(occupied)
    symbol = {k.label: k.symbol for k in sc.kinds}
    legend = [(label if symbol[label] in (label, "") else f"{label} ({symbol[label]})", color[label])
              for label in labels]
    spins = None
    if S is not None and any(k.magnetic for k in sc.kinds):
        magnetic = np.array([k.magnetic for k in sc.kinds])[kind[idx]]
        spins = np.where(magnetic[:, None], S[idx], 0.0)
    return dict(r=(sc.r + u)[idx], rgb=(255 * rgb[kind[idx]]).astype(np.uint8),
                radius=np.full(len(idx), 0.25 * spacing), kind=kind[idx], S=spins, legend=legend)


def add_spheres(plotter, r, radius, rgb=None, values=None, **kwargs):
    """
    Spheres of the given radii (A) as GPU point sprites.

    One shaded splat per atom instead of a triangulated sphere: tens of
    thousands of atoms stay interactive, also on software OpenGL, where
    glyphed spheres (hundreds of triangles each) do not.

    Colors are ``rgb`` (uint8, per atom) or ``values`` mapped through the
    colormap options in ``kwargs`` (``cmap``, ``clim``, ...).
    """
    import pyvista as pv

    cloud = pv.PolyData(np.ascontiguousarray(r, dtype=float))
    cloud["radius"] = np.broadcast_to(np.asarray(radius, dtype=float), (len(cloud.points),)).copy()
    if rgb is not None:
        cloud["rgb"] = rgb
        kwargs.update(scalars="rgb", rgb=True)
    elif values is not None:
        cloud["value"] = values
        kwargs.setdefault("scalars", "value")
    actor = plotter.add_mesh(cloud, style="points_gaussian", render_points_as_spheres=True, emissive=False,
                             **kwargs)
    actor.mapper.SetScaleArray("radius")
    actor.mapper.SetScaleFactor(1.0)
    return actor


def draw_configuration(plotter, style, edges=None, scale=1.0, spins=True, spin_length=None):
    """Draw atoms (sphere sprites) and spin arrows."""
    import pyvista as pv

    r = style["r"]
    add_spheres(plotter, r, style["radius"] * scale, rgb=style["rgb"])
    if spins and style.get("S") is not None:
        S = style["S"]
        keep = np.linalg.norm(S, axis=1) > 0
        if np.any(keep):
            length = spin_length or 3.0 * float(np.median(style["radius"]))
            arrows = pv.PolyData(np.ascontiguousarray(r[keep] - 0.5 * length * S[keep]))
            arrows["S"] = S[keep]
            arrow = pv.Arrow(tip_resolution=8, shaft_resolution=8)
            glyphs = arrows.glyph(orient="S", scale=False, factor=length, geom=arrow)
            plotter.add_mesh(glyphs, color="#3c4043")
    if edges is not None:
        plotter.add_mesh(pv.line_segments_from_points(edges.reshape(-1, 3)), color="black", line_width=1)
    if style.get("legend"):
        width = 0.04 + 0.018 * max(len(label) for label, _ in style["legend"])
        plotter.add_legend([[label, tuple(rgb)] for label, rgb in style["legend"]], bcolor="white",
                           face="circle", loc="upper left",
                           size=(min(0.4, width), 0.045 * len(style["legend"]) + 0.02))
