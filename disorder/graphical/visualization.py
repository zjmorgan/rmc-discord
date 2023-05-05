import numpy as np

from scipy.stats import chi2

import pyvista
pyvista.set_plot_theme('document')

class CrystalStructure:
    """
    Draw a crystal structure. Atom can drawn with various radii with occupancy
    displayed using opacity. Atoms can allso be draw as atomic displacement
    ellipsoids or be decorated with magnetic moments vectors.

    Parameters
    ----------
    A : 2d-array
        Real-space Cartesian transform.
    ux, uy, uz : 1d-array
        Cartesian fractional coordinates.
    atms : 1d-array, str
        Atom labels.
    colors : 2d-array
        RGB color for each atom.
    nu, nv, nw : int
        Number of cells along each dimension. Default is ``1`` along each
        dimension.

    Attributres
    -----------
    pl : plotter
        Plot object.

    Methods
    -------
    show_figure()
    save_figure()
    view_direction)
    draw_basis_vectors()
    draw_cell_edges()
    atomic_radii()
    atomic_displacement_ellipsoids()
    magnetic_vectors()

    """

    def __init__(self, A, ux, uy, uz, atms, colors, nu=1, nv=1, nw=1):

        self.pl = pyvista.Plotter()
        self.pl.enable_parallel_projection()
        self.pl.enable_depth_peeling()
        self.pl.enable_mesh_picking(callback=self.__callback, style='surface',
                                    left_clicking=True, show=True,
                                    show_message=False, smooth_shading=True)

        self._A = A

        self._nu, self._nv, self._nw = nu, nv, nw

        self._ux, self._uy, self._uz = self.__wrap_coordinates(ux, uy, uz)

        self._atms = atms
        self._colors = colors

        self._atm_scale = 0.5

    def __callback(self, mesh):

        display, atom = mesh.__name.split('-')

        i = int(atom)

        n_atm = len(self._colors)

        k = i % n_atm

        atm, ux, uy, uz = self._atms[k], self._ux[i], self._uy[i], self._uz[i]

        u, v, w = np.dot(np.linalg.inv(self._A), [ux,uy,uz])

        header = 'atm        u     v     w\n'
        divide = '========================\n'
        coords = '{:6}{:6.3}{:6.3}{:6.3}\n'

        site = coords.format(atm,u,v,w)

        print(header+divide+site)

    def __wrap_coordinates(self, ux, uy, uz):

        nu, nv, nw = self._nu, self._nv, self._nw

        u, v, w = np.dot(np.linalg.inv(self._A), [ux,uy,uz])

        wrap = np.isclose([u,v,w], 0)
        mask = wrap.sum(axis=0)

        face = mask == 1
        edge = mask == 2
        corner = mask == 3

        indices = np.arange(mask.size)

        face_indices = indices[face]

        u_face, v_face, w_face = u[face].copy(), v[face].copy(), w[face].copy()

        mask_u_face = np.isclose(u_face, 0)
        mask_v_face = np.isclose(v_face, 0)
        mask_w_face = np.isclose(w_face, 0)

        u_face[mask_u_face] += nu
        v_face[mask_v_face] += nv
        w_face[mask_w_face] += nw

        edge_indices = np.repeat(indices[edge], 3)

        u_edge, v_edge, w_edge = u[edge], v[edge], w[edge]

        u_edges = np.repeat(u_edge, 3).reshape(-1,3)
        v_edges = np.repeat(v_edge, 3).reshape(-1,3)
        w_edges = np.repeat(w_edge, 3).reshape(-1,3)

        mask_uv_edge = np.isclose([u_edge,v_edge], 0).all(axis=0)
        mask_vw_edge = np.isclose([v_edge,w_edge], 0).all(axis=0)
        mask_wu_edge = np.isclose([w_edge,u_edge], 0).all(axis=0)

        u_edges[mask_uv_edge] += np.array([nu,0,nu])
        v_edges[mask_uv_edge] += np.array([0,nv,nv])

        v_edges[mask_vw_edge] += np.array([nv,0,nv])
        w_edges[mask_vw_edge] += np.array([0,nw,nw])

        w_edges[mask_wu_edge] += np.array([nw,0,nw])
        u_edges[mask_wu_edge] += np.array([0,nu,nu])

        corner_indices = np.repeat(indices[corner], 7)

        u_corner, v_corner, w_corner = u[corner], v[corner], w[corner]

        u_corners = np.repeat(u_corner, 7).reshape(-1,7)
        v_corners = np.repeat(v_corner, 7).reshape(-1,7)
        w_corners = np.repeat(w_corner, 7).reshape(-1,7)

        mask_uvw_corner = np.isclose([u_corner,
                                      v_corner,
                                      w_corner], 0).all(axis=0)

        u_corners[mask_uvw_corner] += np.array([nu,0,0,nu,nu,0,nu])
        v_corners[mask_uvw_corner] += np.array([0,nv,0,0,nv,nv,nv])
        w_corners[mask_uvw_corner] += np.array([0,0,nw,nw,0,nw,nw])

        u = np.concatenate((u,u_face,u_edges.flatten(),u_corners.flatten()))
        v = np.concatenate((v,v_face,v_edges.flatten(),v_corners.flatten()))
        w = np.concatenate((w,w_face,w_edges.flatten(),w_corners.flatten()))

        self._indices = np.concatenate((indices,face_indices,
                                        edge_indices,corner_indices))

        return np.dot(self._A, [u,v,w])

    def __probability_ellipsoid(self, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy, p=0.99):

        U = np.array([[Uxx,Uxy,Uxz],
                      [Uxy,Uyy,Uyz],
                      [Uxz,Uyz,Uzz]])

        w, v = np.linalg.eig(U)

        r_eff = chi2.ppf(p, 3) if p < 0.999999 else 77.39631549062088

        radii = np.sqrt(r_eff*w.real)

        T = np.dot(v.real,np.dot(np.diag(radii),v.real.T))

        return T

    def show_figure(self):
        """
        Display the drawing.

        """

        self.pl.show()

    def save_figure(self, filename):
        """
        Save the figure as a graphic image.

        Parameters
        ----------
        filename : str
            Name of file with extension. Supported extensions are ``.svg`` and
            ``.pdf``.

        """

        self.pl.save_graphic(filename)

    def view_direction(self, u, v, w):
        """
        View drawing along a crystallographic direction.

        Parameters
        ----------
        u, v, w : float
            Components of the viewing axis. Units are in fractional
            coordinates.

        """

        nu, nv, nw = self._nu, self._nv, self._nw

        x, y, z = 2*np.dot(self._A, [u,v,w])
        xc, yc, zc = np.dot(self._A, [0.5*nu,0.5*nv,0.5*nw])

        self.pl.camera.focal_point = (xc, yc, zc)
        self.pl.camera.position = (xc+x, yc+y, zc+z)

    def draw_basis_vectors(self):
        """
        Draw the baasis vector directions for the crytallographic directions.

        """

        t = self._A.copy()
        t /= np.max(t, axis=1)

        a = pyvista._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                a.SetElement(i,j,t[i,j])

        actor = self.pl.add_axes(xlabel='a', ylabel='b', zlabel='c')
        actor.SetUserMatrix(a)

        actor = self.pl.add_camera_orientation_widget()

        actor.GetRepresentation().GetXPlusLabelProperty()
        actor.GetRepresentation().GetYPlusLabelProperty()
        actor.GetRepresentation().GetZPlusLabelProperty()

    def draw_cell_edges(self):
        """
        Draw unit cell edges.

        """

        T = np.eye(4)
        T[:3,:3] = self._A

        mesh = pyvista.Box(bounds=(0,1,0,1,0,1), level=0, quads=True)
        mesh.transform(T, inplace=True)
        self.pl.add_mesh(mesh, style='wireframe',
                         render_lines_as_tubes=True, show_edges=True)

    def atomic_radii(self, radii, occ):
        """
        Draw atoms with given radii and displayed with opacity corresponding to
        site occupancy

        Parameters
        ----------
        radii : 1d-array
            The radii of the atom sites.
        occ : 1d-array
            The site occupancy of each atom. The value ranges from 0-1.

        """

        n_atm = len(self._colors)

        mesh = pyvista.PolyData(np.column_stack((0,0,0)))

        geom = pyvista.Sphere(radius=1, theta_resolution=60, phi_resolution=60)

        for j, i in enumerate(self._indices):

            k = i % n_atm

            glyph = mesh.glyph(scale=1, geom=geom)

            T = np.zeros((4,4))
            T[0,0] = T[1,1] = T[2,2] = radii[k]*self._atm_scale
            T[:3,-1] = np.array([self._ux[j], self._uy[j], self._uz[j]])
            T[-1,-1] = 1

            glyph.transform(T, inplace=True)

            scalars = np.tile(np.column_stack([*self._colors[k],occ[i]]),
                              geom.n_cells).reshape(-1,4)

            glyph.__name = 's-{}'.format(i)

            self.pl.add_mesh(glyph, name='s-{}'.format(j), scalars=scalars,
                             style='surface', rgb=True, smooth_shading=True)

        self.radii = radii

    def atomic_displacement_ellipsoids(self, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy,
                                       p=0.99):
        """
        Draw atomic displacement ellipsoids

        Parameters
        ----------
        Uxx, Uyy, Uzz, Uyz, Uxz, Uxy : 1d-array
            Atomic displacement parameters in Cartesian coordinates.
        p : float
            Probability surface. Default is ``p=0.99``

        """

        n_atm = len(self._colors)

        mesh = pyvista.PolyData(np.column_stack((0,0,0)))

        geom = pyvista.Sphere(radius=1, theta_resolution=60, phi_resolution=60)

        for j, i in enumerate(self._indices):

            k = i % n_atm

            P = self.__probability_ellipsoid(Uxx[i], Uyy[i], Uzz[i],
                                             Uyz[i], Uxz[i], Uxy[i], p)

            glyph = mesh.glyph(scale=1, geom=geom)

            T = np.zeros((4,4))
            T[:3,:3] = P
            T[:3,-1] = np.array([self._ux[j], self._uy[j], self._uz[j]])
            T[-1,-1] = 1

            glyph.transform(T, inplace=True)

            scalars = np.tile(np.column_stack(self._colors[k]),
                              geom.n_cells).reshape(-1,3)

            glyph.__name = 'e-{}'.format(i)

            self.pl.add_mesh(glyph, name='e-{}'.format(j), scalars=scalars,
                             rgb=True, smooth_shading=True)

    def magnetic_vectors(self, sx, sy, sz):
        """
        Draw magnetic vectors.

        Parameters
        ----------
        sx, sy, sz : 1d-array
            The magnetic moment vector components in Cartesian coordinates.

        """

        n_atm = len(self._colors)

        k = np.arange(len(sx)) % n_atm

        s = np.sqrt(sx**2+sy**2+sz**2)
        mask = s > 0

        mag_scale = 2.5*np.mean(self.radii[k[mask]]/s[mask])*self._atm_scale

        for j, i in enumerate(self._indices):

            r = np.array([self._ux[j],self._uy[j],self._uz[j]])
            v = np.array([sx[i],sy[i],sz[i]])

            s = np.linalg.norm(v)

            if s > 0: v /= s

            glyph = pyvista.Arrow(start=r-s*mag_scale*v,
                                  direction=v, scale=2*s*mag_scale,
                                  shaft_resolution=60, tip_resolution=60)

            self.pl.add_mesh(glyph, color='red',
                             smooth_shading=True, pickable=False)
