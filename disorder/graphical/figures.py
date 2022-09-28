#!/usr/bin/env/python3

import numpy as np

from disorder.graphical.plots import HeatMap, Scatter

def projection(proj, var):

    var = np.array(var)

    coeff = np.abs(proj).round(4).astype(str)

    var[np.isclose(proj,0)] = ''

    coeff[np.isclose(proj,0)] = '0'
    coeff[np.isclose(np.abs(proj),1)] = ''

    sign = ['' if np.isclose(p,0) or p > 0 else '-' for p in proj]

    axis = [s+c+v for s, c, v in zip(sign, coeff, var)]

    return axis

class Intensity3d:

    def __init__(self, canvas, I, extents, B, W):

        self.fig = HeatMap(canvas)

        self.data = I

        h_range, k_range, l_range = extents

        self.min_h, self.max_h = h_range
        self.min_k, self.max_k = k_range
        self.min_l, self.max_l = l_range

        self.nh, self.nk, self.nl = I.shape

        self.proj = np.stack([projection(w, ['h','k','l']) for w in W])
        self.trans = self.__transform(B, W)

    def __transform(self, B, W):

        W = np.array(W)

        return np.linalg.cholesky(np.dot(W.T,np.dot(np.dot(B.T,B),W))).T

    def __value(self, vmin, vmax, size, index):

        if index > size:
            return np.round(vmax, 4)
        elif index < 0 or size <= 1:
            return np.round(vmin, 4)
        else:
            step = (vmax-vmin)/(size-1)
            return np.round(vmin+step*index, 4)

    def __index(self, vmin, vmax, size, value):

        if value > vmax:
            return size-1
        elif value < vmin or size <= 1:
            return 0
        else:
            step = (vmax-vmin)/(size-1)
            return int(round((value-vmin)/step))

    def slice_reciprocal_space(self, hkl, value):

        var = ['h','k','l']

        if hkl == 'h':
            binning = self.min_h, self.max_h, self.nh
            limits = self.min_k, self.min_l, self.max_k, self.max_l
            plane = [1,2]
            axis = 0
        elif hkl == 'k':
            binning = self.min_k, self.max_k, self.nk
            limits = self.min_h, self.min_l, self.max_h, self.max_l
            plane = [0,2]
            axis = 1
        else:
            binning = self.min_l, self.max_l, self.nl
            limits = self.min_h, self.min_k, self.max_h, self.max_k
            plane = [0,1]
            axis = 2

        i_hkl = self.__index(*binning, value)
        value_hkl = self.__value(*binning, i_hkl)

        slice_data = np.take(self.data, i_hkl, axis)
        matrix = np.take(np.take(self.trans, plane, 0), plane, 1)

        W = self.proj

        title = r'$'+var[axis]+'={}$'.format(value_hkl)
        axes = [r'$({{{}}},{{{}}},{{{}}})$'.format(*W[:,p]) for p in plane]

        self.fig.plot_data(slice_data, *limits)
        self.fig.transform_axes(matrix)
        self.fig.set_labels(title, *axes)

        # self.fig.set_normalization(vmin, vmax, norm='linear')
        self.fig.create_colorbar()
        self.fig.set_colorbar_label('r$I(\mathbf{Q})$ [arb. unit]')

class Correlations3d:

    def __init__(self, canvas, data, dx, dy, dz, pairs, A, B):

        self.fig = Scatter(canvas)

        self.dx, self.dy, self.dz, self.pairs = dx, dy, dz, pairs

        self.A = A
        self.B = B

        self.corr, self.corr_sig_sq, *coll = data
        self.coll, self.coll_sig_sq = coll if len(coll) == 2 else (None, None)

    def __mask(self, h, k, l, d=0, tol=1e-4):

        if np.isclose(h**2+k**2+l**2, 0):

            h, k, l = 0, 0, 1

        hx, hy, hz = np.dot(self.B, [h,k,l])

        nx, ny, nz = [hx,hy,hz]/np.linalg.norm([hx,hy,hz])

        Px, Py, Pz = np.cross([0,0,1], [nx,ny,nz])
        P = np.linalg.norm([Px,Py,Pz])

        if np.isclose(P,0):
            Px, Py, Pz = np.cross([0,1,0], [nx,ny,nz])
        elif np.isclose(np.max([Px,Py,Pz]), 0):
            Px, Py, Pz = np.cross([1,0,0], [nx,ny,nz])
        P = np.linalg.norm([Px,Py,Pz])

        px, py, pz = Px/P, Py/P, Pz/P

        Qx, Qy, Qz = np.cross([nx,ny,nz], [px,py,pz])
        Q = np.linalg.norm([Qx,Qy,Qz])

        qx, qy, qz = Qx/Q, Qy/Q, Qz/Q

        plane = np.isclose(hx*self.dx+hy*self.dy+hz*self.dz, d, rtol=tol)

        A_inv = np.linalg.inv(self.A)

        pu, pv, pw = np.dot(A_inv, [px,py,pz])
        qu, qv, qw = np.dot(A_inv, [qx,qy,qz])

        proj_x = np.array([pu,pv,pw])
        proj_y = np.array([qu,qv,qw])

        scale_dx = proj_x.max()
        scale_dy = proj_y.max()

        proj_x /= scale_dx
        proj_y /= scale_dy

        aspect = scale_dx/scale_dy

        Dx = (px*self.dx[plane]+py*self.dy[plane]+pz*self.dz[plane])*scale_dx
        Dy = (qx*self.dx[plane]+qy*self.dy[plane]+qz*self.dz[plane])*scale_dy

        return aspect, proj_x, proj_y, Dx, Dy

    def slice_real_space(self, h, k, l, d=0, tol=1e-4, dataset='correlation'):

        var = ['u','v','w']

        aspect, proj_x, proj_y, Dx, Dy = self.__mask(h, k, l, d, tol)

        if dataset == 'collinearity' and self.coll is not None:
            data = self.coll
            label = r'$C_2(\mathbf{r}-\mathbf{r^\prime})$'
            colormap = 'binary'
            vmin = 0
        else:
            data = self.corr
            label = r'$C_1(\mathbf{r}-\mathbf{r^\prime})$'
            colormap = 'diverging'
            vmin = -1

        hkl = np.array([h,k,l])
        scale = np.gcd.reduce(hkl)

        if scale != 0:
            hkl //= scale

        hkl = ['\bar{{'+str(val)+'}}' if val < 0 else str(val) for val in hkl]

        d = np.round(d, int(-np.log10(tol)))

        title = r'$({}{}{})\cdot[uvw]={}$'.format(*hkl,d)

        px, py = [projection(p, var) for p in (proj_x,proj_y)]

        axes = [r'$[{{{}}},{{{}}},{{{}}}]$'.format(*proj) for proj in (px,py)]

        self.fig.plot_data(Dx, Dy, data)
        self.fig.set_labels(title, *axes)

        self.fig.update_colormap(colormap)
        self.fig.create_colorbar(norm='symlog')

        self.fig.set_normalization(vmin, 1, norm='symlog')
        self.fig.set_aspect(aspect)

        self.fig.set_colorbar_label(label)
        self.fig.draw_canvas()
