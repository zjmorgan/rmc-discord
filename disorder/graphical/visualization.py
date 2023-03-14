import numpy as np

from scipy.stats import chi2

import pyvista
pyvista.set_plot_theme('document')

class CrystalStructure:

    def __init__(self, A, ux, uy, uz, atms, colors, nu=1, nv=1, nw=1):

        self.pl = pyvista.Plotter()
        self.pl.enable_parallel_projection()
        self.pl.enable_depth_peeling()
        self.pl.enable_mesh_picking(callback=self.callback, style='surface',
                                    left_clicking=True, show=True,
                                    show_message=False, smooth_shading=True)

        self.A = A

        self.nu, self.nv, self.nw = nu, nv, nw

        self.ux, self.uy, self.uz = self.__wrap_coordinates(ux, uy, uz)

        self.atms = atms
        self.colors = colors

        self.atm_scale = 0.5

    def callback(self, mesh):

        display, atom = mesh.__name.split('-')

        i = int(atom)

        n_atm = len(self.colors)

        k = i % n_atm

        atm, ux, uy, uz = self.atms[k], self.ux[i], self.uy[i], self.uz[i]

        u, v, w = np.dot(np.linalg.inv(self.A), [ux,uy,uz])

        header = 'atm        u     v     w\n'
        divide = '========================\n'
        coords = '{:6}{:6.3}{:6.3}{:6.3}\n'

        site = coords.format(atm,u,v,w)

        print(header+divide+site)

    def __wrap_coordinates(self, ux, uy, uz):

        nu, nv, nw = self.nu, self.nv, self.nw

        u, v, w = np.dot(np.linalg.inv(self.A), [ux,uy,uz])

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

        self.indices = np.concatenate((indices,face_indices,
                                       edge_indices,corner_indices))

        return np.dot(self.A, [u,v,w])

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

        self.pl.show()

    def save_figure(self, filename):

        self.pl.save_graphic(filename)

    def view_direction(self, u, v, w):

        nu, nv, nw = self.nu, self.nv, self.nw

        x, y, z = 2*np.dot(self.A, [u,v,w])
        xc, yc, zc = np.dot(self.A, [0.5*nu,0.5*nv,0.5*nw])

        self.pl.camera.focal_point = (xc, yc, zc)
        self.pl.camera.position = (xc+x, yc+y, zc+z)

    def draw_basis_vectors(self):

        t = self.A.copy()
        t /= np.max(t, axis=1)

        a = pyvista._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                a.SetElement(i,j,t[i,j])

        actor = self.pl.add_axes(xlabel='a', ylabel='b', zlabel='c')

        actor.SetUserMatrix(a)

    def draw_cell_edges(self):

        uc = np.array([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0])
        vc = np.array([0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0])
        wc = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0])

        ucx, ucy, ucz = np.dot(self.A, [uc,vc,wc])

        connections = ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
                       (2,6),(3,7),(4,5),(4,6),(5,7),(6,7))

        for i, j in connections:
            points = np.row_stack(((ucx[i],ucy[i],ucz[i]),
                                   (ucx[j],ucy[j],ucz[j])))
            self.pl.add_lines(points, color='black', width=2)

    def atomic_radii(self, radii, occ):

        n_atm = len(self.colors)

        for j, i in enumerate(self.indices):

            k = i % n_atm

            mesh = pyvista.PolyData(np.column_stack((0,0,0)))

            geom = pyvista.Sphere(radius=1, theta_resolution=60,
                                  phi_resolution=60)

            glyph = mesh.glyph(scale=1, geom=geom)

            T = np.zeros((4,4))
            T[0,0] = T[1,1] = T[2,2] = radii[k]*self.atm_scale
            T[:3,-1] = np.array([self.ux[j], self.uy[j], self.uz[j]])
            T[-1,-1] = 1

            glyph.transform(T, inplace=True)

            scalars = np.tile(np.column_stack([*self.colors[k],occ[i]]),
                              geom.n_cells).reshape(-1,4)

            glyph.__name = 's-{}'.format(i)

            self.pl.add_mesh(glyph, name='s-{}'.format(j), scalars=scalars,
                             style='surface', rgb=True, smooth_shading=True)

        self.radii = radii

    def atomic_displacement_ellipsoids(self, Uxx, Uyy, Uzz, Uyz, Uxz, Uxy,
                                       p=0.99):

        n_atm = len(self.colors)

        for j, i in enumerate(self.indices):

            k = i % n_atm

            P = self.__probability_ellipsoid(Uxx[i], Uyy[i], Uzz[i],
                                             Uyz[i], Uxz[i], Uxy[i], p)

            mesh = pyvista.PolyData(np.column_stack((0,0,0)))

            geom = pyvista.Sphere(radius=1, theta_resolution=60,
                                  phi_resolution=60)

            glyph = mesh.glyph(scale=1, geom=geom)

            T = np.zeros((4,4))
            T[:3,:3] = P
            T[:3,-1] = np.array([self.ux[j], self.uy[j], self.uz[j]])
            T[-1,-1] = 1

            glyph.transform(T, inplace=True)

            scalars = np.tile(np.column_stack(self.colors[k]),
                              geom.n_cells).reshape(-1,3)

            glyph.__name = 'e-{}'.format(i)

            self.pl.add_mesh(glyph, name='e-{}'.format(j), scalars=scalars,
                             rgb=True, smooth_shading=True)

    def magnetic_vectors(self, sx, sy, sz):

        n_atm = len(self.colors)

        k = np.arange(len(sx)) % n_atm

        s = np.sqrt(sx**2+sy**2+sz**2)
        mask = s > 0

        mag_scale = 2.5*np.mean(self.radii[k[mask]]/s[mask])*self.atm_scale

        for j, i in enumerate(self.indices):

            r = np.array([self.ux[j],self.uy[j],self.uz[j]])
            v = np.array([sx[i],sy[i],sz[i]])

            s = np.linalg.norm(v)

            if s > 0:
                v /= s

            glyph = pyvista.Arrow(start=r-s*mag_scale*v,
                                  direction=v, scale=2*s*mag_scale,
                                  shaft_resolution=60, tip_resolution=60)

            self.pl.add_mesh(glyph, color='red',
                             smooth_shading=True, pickable=False)
