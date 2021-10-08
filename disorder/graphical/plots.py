#!/usr/bin/env/python3

import re
import numpy as np

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

from matplotlib import ticker
from matplotlib.ticker import Locator

from itertools import cycle

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
matplotlib.rcParams['mathtext.cal'] = 'sans'
matplotlib.rcParams['mathtext.rm'] = 'sans'
matplotlib.rcParams['mathtext.sf'] = 'sans'
matplotlib.rcParams['mathtext.tt'] = 'monospace'

matplotlib.rcParams['axes.titlesize'] = 'medium'
matplotlib.rcParams['axes.labelsize'] = 'medium'

matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['legend.loc'] = 'best'
matplotlib.rcParams['legend.fontsize'] = 'medium'

class MinorSymLogLocator(Locator):
    
    def __init__(self, linthresh, nints=10):
        
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        dmlower = majorlocs[1]-majorlocs[0]
        dmupper = majorlocs[-1]-majorlocs[-2]

        if (majorlocs[0] != 0. and 
            ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or 
             (dmlower == self.linthresh and majorlocs[0] < 0))):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        if (majorlocs[-1] != 0. and 
            ((np.abs(majorlocs[-1]) != self.linthresh 
              and dmupper > self.linthresh) or 
             (dmupper == self.linthresh and majorlocs[-1] > 0))):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        minorlocs = []

        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i]-majorlocs[i-1]
            if abs(majorlocs[i-1]+majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals-1.

            minorstep = majorstep/ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

class Plot():
    
    def __init__(self, canvas):
        
        self.canvas = canvas
        self.fig = canvas.figure
        self.ax = canvas.figure.add_subplot(111)
        self.ax.minorticks_on()
        
    def save_figure(self, filename):
        
        self.fig.savefig(filename)
        
    def clear_canvas(self):
        
        self.ax.remove()
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.minorticks_on()
        
    def draw_canvas(self):
        
        self.canvas.draw_idle()
        
    def tight_layout(self, pad=3.24):
        
        self.fig.tight_layout(pad=pad)
        
    def set_labels(self, title='', xlabel='', ylabel=''):
        
        self.ax.set_title(title) 
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
                
class Line(Plot):
    
    def __init__(self, canvas):
        
        super(Line, self).__init__(canvas)
        
        self.p = []
        
        self.hl = None
        self.twin_ax = None
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        prop_cycle = matplotlib.rcParams['axes.prop_cycle']
        self.colors = cycle(prop_cycle.by_key()['color'])
        
    def __get_axis(self, twin=False):
        
        if not twin:
            return self.ax
        else:
            if self.twin_ax is None:
                self.twin_ax = self.ax.twinx()
            return self.twin_ax
                
    def set_labels(self, title='', xlabel='', ylabel='', twin_ylabel=''):
        
        super().set_labels(title=title, xlabel=xlabel, ylabel=ylabel)
        
        if self.twin_ax:
            self.twin_ax.set_ylabel(twin_ylabel) 
        
    def clear_canvas(self):
        
        super().clear_canvas()
        self.clear_lines()
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
    def clear_lines(self):
        
        if self.p:
            for p in self.p:
                p.lines[0].remove()
            self.p = []
            
        if self.hl: 
            self.hl.remove()
            self.hl = None
        if self.twin_ax: 
            self.twin_ax.remove()
            self.twin_ax = None
        
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        self.set_labels()
        
        prop_cycle = matplotlib.rcParams['axes.prop_cycle']
        self.colors = cycle(prop_cycle.by_key()['color'])
        
    def set_normalization(self, norm='linear', twin=False):
        
        ax = self.__get_axis(twin)
        
        if (norm.lower() == 'linear'): 
            ax.set_yscale('linear')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        elif (norm.lower() == 'logarithmic'):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            # ax.yaxis.set_minor_locator(ticker.LogLocator())
        else:
            thresh, scale = 0.1, 0.9
            ax.set_yscale('symlog', linthresh=thresh, linscale=scale)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_locator(MinorSymLogLocator(0.1))
            
    def set_limts(self, vmin=None, vmax=None, twin=False):
                
        xmin, xmax, ymin, ymax = self.ax.axis()
                
        if vmin is not None: ymin = vmin
        if vmax is not None: ymax = vmax
        
        margin = 0.05
        
        ax = self.__get_axis(twin)
    
        transform = ax.yaxis.get_transform()
        inverse_trans = transform.inverted()

        ymint, ymaxt = transform.transform([ymin,ymax])
        delta = (ymaxt-ymint)*margin

        ymin, ymax = inverse_trans.transform([ymint-delta,ymaxt+delta])
        
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
            
    def reset_view(self, twin=False):
    
        ax = self.__get_axis(twin)
        
        ax.relim()    
        ax.autoscale_view()
        
    def update_data(self, x, y, i=0):
            
        self.p[i].lines[0].set_data(x, y)
        
    def get_data(self, i=0):
            
        return self.p[i].lines[0].get_xydata().T
        
    def plot_data(self, x, y, yerr=None, marker='o', label='', twin=False):
        
        ax = self.__get_axis(twin)
        
        color = next(self.colors)
                
        err = ax.errorbar(x, y, yerr=yerr, fmt=marker, label=label,
                          color=color, ecolor=color, clip_on=False)
        
        self.p.append(err)
            
    def show_legend(self):
                                        
        handles = [p for p in self.p if p.get_label() != '']
        labels = [p.get_label() for p in handles]
        self.ax.legend(handles, labels)
        
    def draw_horizontal(self):
        
        self.hl = self.ax.axhline(y=0, xmin=0, xmax=1, color='k', \
                                  linestyle='-', linewidth=0.8)
            
    def use_scientific(self, twin=False):
    
        ax = self.__get_axis(twin)
        
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
            
class HeatMap(Plot):
            
    def __init__(self, canvas):
        
        super(HeatMap, self).__init__(canvas)
        
        self.im = None
        self.norm = None
        self.__color_limits()
        
    def __matrix_transform(self, matrix):
            
        matrix /= matrix[1,1]
        
        scale = 1/matrix[0,0]
        matrix[0,:] *= scale
                
        return matrix, scale
        
    def __extents(self, min_x, min_y, max_x, max_y, size_x, size_y):
            
        dx = 0 if size_x <= 1 else (max_x-min_x)/(size_x-1)
        dy = 0 if size_y <= 1 else (max_y-min_y)/(size_y-1)
                
        return [min_x-dx/2, max_x+dx/2, min_y-dy/2, max_y+dy/2]
            
    def __reverse_extents(self):
            
        extents = self.im.get_extent()
        size_x, size_y = self.im.get_size()
        
        range_x = extents[1]-extents[0]
        range_y = extents[3]-extents[2]
        
        dx = 0 if size_x <= 1 else range_x/(size_x-1)
        dy = 0 if size_y <= 1 else range_y/(size_y-1)
        
        min_x = extents[0]+dx/2
        max_x = extents[1]-dx/2
        
        min_y = extents[2]+dy/2
        max_y = extents[3]-dy/2
        
        return [min_x, max_x, min_y, max_y]    

    def __transform_extents(self, matrix, extents):
    
        ext_min = np.dot(matrix, extents[0::2])
        ext_max = np.dot(matrix, extents[1::2])
        
        return ext_min, ext_max
    
    def __offset(self, matrix, minimum):
        
        return -np.dot(matrix, [0,minimum])[0]
            
    def __color_limits(self, category='sequential'):
        
        if (category.lower() == 'sequential'):
            self.cmap = plt.cm.viridis
        elif (category.lower() == 'diverging'):
            self.cmap = plt.cm.bwr
        else:
            self.cmap = plt.cm.binary
    
    def set_normalization(self, vmin, vmax, norm='linear'):
        
        if np.isclose(vmin, vmax): vmin, vmax = 1e-3, 1e+3
        
        if (norm.lower() == 'linear'):
            self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
        elif (norm.lower() == 'logarithmic'):
            self.norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.norm = colors.SymLogNorm(linthresh=0.1, linscale=0.9, base=10, 
                                          vmin=vmin, vmax=vmax)
            
        if self.im.colorbar is not None:
            
            orientation = self.im.colorbar.orientation

            self.remove_colorbar()
            self.create_colorbar(orientation, norm)
            
    def update_normalization(self, norm='linear'):

        if self.im is not None:

            vmin, vmax = self.im.get_clim()
            
            self.set_normalization(vmin, vmax, norm)
                
            self.im.set_norm(self.norm)
            
    def update_colormap(self, category='sequential'):
            
        self.__color_limits(category)
        
        if self.im is not None:
            
            self.im.set_cmap(self.cmap)
            
    def create_colorbar(self, orientation='vertical', norm='linear'):
        
        self.remove_colorbar()
        
        pad = 0.05 if orientation.lower() == 'vertical' else 0.2
        
        self.cb = self.fig.colorbar(self.im, ax=self.ax, 
                                    orientation=orientation, pad=pad)
        
        self.cb.ax.minorticks_on()
    
        if (norm.lower() == 'linear'):
            self.cb.formatter.set_powerlimits((0, 0))
            self.cb.update_ticks()
            
    def remove_colorbar(self):
        
        if self.im.colorbar is not None:
            
            self.im.colorbar.remove()
            
    def reset_color_limits(self):
            
        self.im.autoscale()
                    
    def update_data(self, data, vmin, vmax):
        
        if self.im is not None:
            
            self.im.set_array(data.T)
            self.im.set_clim(vmin=vmin, vmax=vmax)
            
    def get_data(self):
        
        if self.im is not None:
            
            return self.im.get_array().T
                    
    def plot_data(self, data, min_x, min_y, max_x, max_y, matrix=np.eye(2)):
        
        size_x, size_y = data.shape[1], data.shape[0]
        
        extents = self.__extents(min_x, min_y, max_x, max_y, size_x, size_y)
        
        self.im = self.ax.imshow(data.T, interpolation='nearest', 
                                 origin='lower', extent=extents)
        
        self.transform_axes(matrix)
        
        self.ax.minorticks_on()
        
    def transform_axes(self, matrix):
                            
        extents = self.__reverse_extents()
        transformation, scale = self.__matrix_transform(matrix)
        ext_min, ext_max = self.__transform_extents(transformation, extents)
        
        min_y = extents[2]
        offset = self.__offset(transformation, min_y)
    
        trans = mtransforms.Affine2D()
                
        trans_matrix = np.eye(3)
        trans_matrix[0:2,0:2] = transformation
        trans.set_matrix(trans_matrix)
        
        shift = mtransforms.Affine2D().translate(offset,0)
        
        self.ax.set_aspect(scale)
        
        trans_data = trans+shift+self.ax.transData
        self.im.set_transform(trans_data)
        
        self.ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
        self.ax.set_ylim(ext_min[1],ext_max[1])
    
class Scatter(Plot):
    
    def __init__(self, canvas):
        
        super(Scatter, self).__init__(canvas)
        
        self.s = None
        self.__color_limits()
        
    def __mask_plane(self, dx, dy, dz, h, k, l, d, A, B, tol):
                 
         hx, hy, hz = np.dot(B, [h,k,l])
             
         if (not np.isclose(hx**2+hy**2+hz**2,0)):
             
             nx, ny, nz = [hx,hy,hz]/np.linalg.norm([hx,hy,hz])
             
             Px, Py, Pz = np.cross([0,0,1], [nx,ny,nz])
             P = np.linalg.norm([Px,Py,Pz])
             
             if (np.isclose(P,0)):
                 Px, Py, Pz = np.cross([0,1,0], [nx,ny,nz])
                 P = np.linalg.norm([Px,Py,Pz])            
             elif (np.isclose(np.max([Px,Py,Pz]),0)):
                 Px, Py, Pz = np.cross([1,0,0], [nx,ny,nz])
                 P = np.linalg.norm([Px,Py,Pz])
                 
             px, py, pz = Px/P, Py/P, Pz/P
     
             Qx, Qy, Qz = np.cross([nx,ny,nz], [px,py,pz])
             Q = np.linalg.norm([Qx,Qy,Qz])                          
     
             qx, qy, qz = Qx/Q, Qy/Q, Qz/Q
    
             plane = np.isclose(hx*dx+hy*dy+hz*dz, d, rtol=tol)
                      
             A_inv = np.linalg.inv(A)
              
             pu, pv, pw = np.dot(A_inv, [px,py,pz])
             qu, qv, qw = np.dot(A_inv, [qx,qy,qz])
             
             projx = np.array([pu,pv,pw])
             projy = np.array([qu,qv,qw])
                                     
             scale_dx = projx.max()
             scale_dy = projy.max()
             
             projx = projx/scale_dx
             projy = projy/scale_dy
             
             cor_aspect = scale_dx/scale_dy
           
             dx = (px*dx[plane]+py*dy[plane]+pz*dz[plane])*scale_dx
             dy = (qx*dx[plane]+qy*dy[plane]+qz*dz[plane])*scale_dy
             
             return cor_aspect, projx, projy, dx, dy, plane

    def __color_limits(self, category='sequential'):
        
        if (category == 'sequential'):
            self.cmap = plt.cm.viridis
        elif (category == 'diverging'):
            self.cmap = plt.cm.bwr
        else:
            self.cmap = plt.cm.binary
                
    def set_normalization(self, vmin, vmax, norm='linear'):
        
        if np.isclose(vmin, vmax): vmin, vmax = 1e-3, 1e+3
        
        if (norm.lower() == 'linear'):
            self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
        elif (norm.lower() == 'logarithmic'):
            self.norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.norm = colors.SymLogNorm(linthresh=0.1, linscale=0.9, base=10, 
                                          vmin=vmin, vmax=vmax)
            
        if self.s.colorbar is not None:
            
            orientation = self.s.colorbar.orientation

            self.remove_colorbar()
            self.create_colorbar(orientation, norm)
            
    def update_normalization(self, norm='linear'):
        
        if self.s is not None:

            vmin, vmax = self.s.get_clim()
        
            self.set_normalization(vmin, vmax, norm)
                
            self.s.set_norm(self.norm)
            
    def update_colormap(self, category='sequential'):
            
        self.__color_limits(category)
        
        if self.s is not None:
            
            self.s.set_cmap(self.cmap)
            
    def create_colorbar(self, orientation='vertical', norm='linear'):
        
        if self.s is not None:
            
            self.remove_colorbar()
            
            pad = 0.05 if orientation.lower() == 'vertical' else 0.2
            
            self.cb = self.fig.colorbar(self.s, ax=self.ax, 
                                        orientation=orientation, pad=pad)
            
            self.cb.ax.minorticks_on()
        
            if (norm.lower() == 'linear'):
                self.cb.formatter.set_powerlimits((0, 0))
                self.cb.update_ticks()
            
    def remove_colorbar(self):
        
        if self.s.colorbar is not None:
            
            self.s.colorbar.remove()
            
    def set_colorbar_label(self, label):
        
        if self.s.colorbar is not None:

            self.s.colorbar.set_label(label)
            
    def reset_color_limits(self):
            
        if self.s is not None:
            
            self.s.autoscale()
                    
    def update_data(self, c, vmin, vmax):
        
        if self.s is not None:
            
            self.s.set_array(c)
            self.s.set_clim(vmin=vmin, vmax=vmax)
            
    def get_data(self):
        
        if self.s is not None:
            
            return self.s.get_array()
                    
    def plot_data(self, x, y, c):
        
        self.s = self.ax.scatter(x, y, c=c, cmap=self.cmap)
            
    def update_normalization(self, norm):
        
        self.set_normalization(norm)
    
        if self.s is not None:
            
            self.s.set_cmap(self.cmap)
            self.s.set_norm(self.norm)

def __matrix_transform(B, layer='l', T=np.eye(3)):

    matrix = np.eye(3)
    
    Bp = np.linalg.cholesky(np.dot(T.T,np.dot(np.dot(B.T,B),T))).T
    
    if (layer == 'h'):
        Q = Bp[1:3,1:3].copy()
    elif (layer == 'k'):
        Q = Bp[0:3:2,0:3:2].copy()
    elif (layer == 'l'):
        Q = Bp[0:2,0:2].copy()     
                   
    Q /= Q[1,1]
    
    scale = 1/Q[0,0]
    Q[0,:] *= scale
    
    matrix[0:2,0:2] = Q
    
    return matrix, scale
    
def __extents(min_x, min_y, max_x, max_y, size_x, size_y):
        
    dx = 0 if size_x <= 1 else (max_x-min_x)/(size_x-1)
    dy = 0 if size_y <= 1 else (max_y-min_y)/(size_y-1)
            
    return [min_x-dx/2, max_x+dx/2, min_y-dy/2, max_y+dy/2]
        
def __transform_extents(matrix, extents):

    ext_min = np.dot(matrix[0:2,0:2], extents[0::2])
    ext_max = np.dot(matrix[0:2,0:2], extents[1::2])
    
    return ext_min, ext_max

def __offset(matrix, minimum):
    
    return -np.dot(matrix, [0,minimum,0])[0]

def plot_exp_h(canvas, data, h, ih, min_k, min_l, max_k, max_l, size_k, size_l, 
               matrix_h, scale_h, norm, vmin, vmax):
        
    extents_h = [min_k, max_k, min_l, max_l]
    
    ext_min_h, ext_max_h = __transform_extents(matrix_h, extents_h)
    
    extents_h = __extents(min_k, min_l, max_k, max_l, size_k, size_l)
    
    offset_h = __offset(matrix_h, min_l)

    if (norm == 'Logarithmic'):
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
        
    fig = canvas.figure
    fig.clear()   
    
    ax_h = fig.add_subplot(111)
    
    im_h = ax_h.imshow(data[ih,:,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_h,
                       zorder=100)
    
    trans_h = mtransforms.Affine2D()
    
    trans_h.set_matrix(matrix_h)
    
    shift_h = mtransforms.Affine2D().translate(offset_h,0)
    
    ax_h.set_aspect(scale_h)
    
    trans_data_h = trans_h+shift_h+ax_h.transData
    
    im_h.set_transform(trans_data_h)
    
    ax_h.set_xlim(ext_min_h[0]+offset_h,ext_max_h[0]+offset_h)
    ax_h.set_ylim(ext_min_h[1],ext_max_h[1])
            
    ax_h.xaxis.tick_bottom()
    
    ax_h.axes.tick_params(labelsize='small')
    
    ax_h.set_title(r'$h={}$'.format(h), fontsize='small') 
    
    ax_h.set_xlabel(r'$(0k0)$', fontsize='small')
    ax_h.set_ylabel(r'$(00l)$', fontsize='small')
    
    ax_h.minorticks_on()
    
    fig.tight_layout(pad=3.24)
    
    cb = fig.colorbar(im_h, ax=ax_h, orientation='horizontal', pad=0.2)
    cb.ax.minorticks_on()
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    canvas.draw()
    
def plot_exp_k(canvas, data, k, ik, min_h, min_l, max_h, max_l, size_h, size_l, 
               matrix_k, scale_k, norm, vmin, vmax):
    
    extents_k = [min_h, max_h, min_l, max_l]
    
    ext_min_k, ext_max_k = __transform_extents(matrix_k, extents_k)
    
    extents_k = __extents(min_h, min_l, max_h, max_l, size_h, size_l)
    
    offset_k = __offset(matrix_k, min_l)

    if (norm == 'Logarithmic'):
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
        
    fig = canvas.figure
    fig.clear()   
    
    ax_k = fig.add_subplot(111)
    
    im_k = ax_k.imshow(data[:,ik,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_k,
                       zorder=100)
    
    trans_k = mtransforms.Affine2D()
    
    trans_k.set_matrix(matrix_k)
    
    shift_k = mtransforms.Affine2D().translate(offset_k,0)
    
    ax_k.set_aspect(scale_k)
    
    trans_data_k = trans_k+shift_k+ax_k.transData
    
    im_k.set_transform(trans_data_k)
    
    ax_k.set_xlim(ext_min_k[0]+offset_k,ext_max_k[0]+offset_k)
    ax_k.set_ylim(ext_min_k[1],ext_max_k[1])
            
    ax_k.xaxis.tick_bottom()
    
    ax_k.axes.tick_params(labelsize='small')
    
    ax_k.set_title(r'$k={}$'.format(k), fontsize='small') 
    
    ax_k.set_xlabel(r'$(h00)$', fontsize='small')
    ax_k.set_ylabel(r'$(00l)$', fontsize='small')
    
    ax_k.minorticks_on()
    
    fig.tight_layout(pad=3.24)
    
    cb = fig.colorbar(im_k, ax=ax_k, orientation='horizontal', pad=0.2)
    cb.ax.minorticks_on()
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    canvas.draw()

def plot_exp_l(canvas, data, l, il, min_h, min_k, max_h, max_k, size_h, size_k, 
               matrix_l, scale_l, norm, vmin, vmax):
    
    extents_l = [min_h, max_h, min_k, max_k]
    
    ext_min_l, ext_max_l = __transform_extents(matrix_l, extents_l)
    
    extents_l = __extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_l = __offset(matrix_l, min_k)
    
    if (norm == 'Logarithmic'):
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)

    fig = canvas.figure
    fig.clear()   
    
    ax_l = fig.add_subplot(111)
    
    im_l = ax_l.imshow(data[:,:,il].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_l,
                       zorder=100)
    
    trans_l = mtransforms.Affine2D()
    
    trans_l.set_matrix(matrix_l)
    
    shift_l = mtransforms.Affine2D().translate(offset_l,0)
    
    ax_l.set_aspect(scale_l)
    
    trans_data_l = trans_l+shift_l+ax_l.transData
    
    im_l.set_transform(trans_data_l)
    
    ax_l.set_xlim(ext_min_l[0]+offset_l,ext_max_l[0]+offset_l)
    ax_l.set_ylim(ext_min_l[1],ext_max_l[1])
            
    ax_l.xaxis.tick_bottom()
    
    ax_l.axes.tick_params(labelsize='small')
    
    ax_l.set_title(r'$l={}$'.format(l), fontsize='small') 
    
    ax_l.set_xlabel(r'$(h00)$', fontsize='small') 
    ax_l.set_ylabel(r'$(0k0)$', fontsize='small') 
    
    ax_l.minorticks_on()
    
    fig.tight_layout(pad=3.24)
 
    cb = fig.colorbar(im_l, ax=ax_l, orientation='horizontal', pad=0.2)
    cb.ax.minorticks_on()
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        
    canvas.draw()
    
def plot_ref(canvas, data, hkl, slice_hkl, i_hkl, 
             min_h, min_k, min_l, max_h, max_k, max_l, size_h, size_k, size_l, 
             matrix_h, matrix_k, matrix_l, scale_h, scale_k, scale_l,
             norm, vmin, vmax):
    
    extents_h = [min_k, max_k, min_l, max_l]
    extents_k = [min_h, max_h, min_l, max_l]
    extents_l = [min_h, max_h, min_k, max_k]
    
    ext_min_h, ext_max_h = __transform_extents(matrix_h, extents_h)
    ext_min_k, ext_max_k = __transform_extents(matrix_k, extents_k)
    ext_min_l, ext_max_l = __transform_extents(matrix_l, extents_l)
    
    extents_h = __extents(min_k, min_l, max_k, max_l, size_k, size_l)
    extents_k = __extents(min_h, min_l, max_h, max_l, size_h, size_l)
    extents_l = __extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_h = __offset(matrix_h, min_l)
    offset_k = __offset(matrix_k, min_l)
    offset_l = __offset(matrix_l, min_k)
    
    if (norm == 'Logarithmic'):
        if (np.isclose(vmin, 0) and np.isclose(vmax, 0)):
            vmin, vmax = 1e-3, 1e-2
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
                
    fig = canvas.figure
    fig.clear()   
    
    ax = fig.add_subplot(111)
    
    ax.set_aspect(1.)
    
    if (hkl == 'h ='):
    
        im = ax.imshow(data[i_hkl,:,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_h)
        
        ax.set_title(r'$h={}$'.format(slice_hkl), fontsize='small') 

        ax.set_xlabel(r'$(0k0)$', fontsize='small')
        ax.set_ylabel(r'$(00l)$', fontsize='small')
         
        matrix, offset, scale = matrix_h, offset_h, scale_h
        ext_min, ext_max = ext_min_h, ext_max_h
        
    elif (hkl == 'k ='):
        
        im = ax.imshow(data[:,i_hkl,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_k)
    
        ax.set_title(r'$k={}$'.format(slice_hkl), fontsize='small')
        
        ax.set_xlabel(r'$(h00)$', fontsize='small')
        ax.set_ylabel(r'$(00l)$', fontsize='small')
        
        matrix, offset, scale = matrix_k, offset_k, scale_k
        ext_min, ext_max = ext_min_k, ext_max_k
        
    else:
    
        im = ax.imshow(data[:,:,i_hkl].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_l)
        
        ax.set_title(r'$l={}$'.format(slice_hkl), fontsize='small') 
    
        ax.set_xlabel(r'$(h00)$', fontsize='small') 
        ax.set_ylabel(r'$(0k0)$', fontsize='small') 
        
        matrix, offset, scale = matrix_l, offset_l, scale_l
        ext_min, ext_max = ext_min_l, ext_max_l
   
    trans = mtransforms.Affine2D()
    
    trans.set_matrix(matrix)
            
    shift = mtransforms.Affine2D().translate(offset,0)
    
    ax.set_aspect(scale)
    
    trans_data = trans+shift+ax.transData
    
    im.set_transform(trans_data)
    
    ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
    ax.set_ylim(ext_min[1],ext_max[1])   

    ax.xaxis.tick_bottom()
    
    ax.minorticks_on()
    
    ax.axes.tick_params(labelsize='small')

    fig.tight_layout(pad=3.24)
    
    cb = fig.colorbar(im, ax=ax)
    cb.ax.minorticks_on()
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        
    cb.ax.tick_params(labelsize='small') 
    canvas.draw()
    
    return im
    
def fast_update_ref(canvas, im, data, hkl, i_hkl, vmin, vmax):
   
    if (hkl == 'h ='):
        im.set_array(data[i_hkl,:,:].T)
    elif (hkl == 'k ='):
        im.set_array(data[:,i_hkl,:].T)        
    else:
        im.set_array(data[:,:,i_hkl].T) 
        
    im.set_clim(vmin=vmin, vmax=vmax)

    canvas.draw_idle()           
    
def correlations_3d(canvas, dx, dy, dz, h, k, l, d, data, error, atm_pair3d, 
                    disorder, correlation, average, norm, atoms, pairs, 
                    A, B, tol):
    
    aspect, proj0, proj1, d0, d1, plane = __mask_plane(dx, dy, dz, h, k, l, 
                                                      d, A, B, tol)

    if (correlation == 'Correlation'):
        vmin = -1.0
        cmap = plt.cm.bwr
    else:
        vmin = 0.0
        cmap = plt.cm.binary
        
    if (norm == 'Logarithmic'):
        normalize = colors.SymLogNorm(linthresh=0.1, 
                                      linscale=1.0*(1-10**-1),
                                      base=10, vmin=vmin, vmax=1.0)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=1.0)
                
    fig = canvas.figure
    fig.clear()   
    
    ax = fig.add_subplot(111)
            
    if average:   
        
        s = ax.scatter(d0, d1, c=data[plane], norm=normalize, cmap=cmap)
        
    else:
        
        for atom, pair in zip(atoms, pairs):
            
            if (atom == 'self-correlation'):
                mask = atm_pair3d[plane] == '0'
            else:
                mask = atm_pair3d[plane] == atom+'_'+pair
                
            s = ax.scatter(d0[mask], d1[mask], c=data[plane][mask], 
                           norm=normalize, cmap=cmap)
        
    if (len(atoms) == 0): s = ax.scatter(0, 0, c=0, norm=normalize, cmap=cmap)            

    cb = fig.colorbar(s, format='%.1f')
    cb.ax.minorticks_on()
    
    if (norm == 'Logarithmic'):
        cb.locator = ticker.SymmetricalLogLocator(linthresh=0.1, base=10)
        cb.update_ticks()
        if (correlation == 'Correlation'):
            minorticks = np.concatenate((s.norm(np.linspace(-1, -0.1, 11)), 
                                         s.norm(np.linspace(-0.1, 0.1, 21)), 
                                         s.norm(np.linspace(0.1, 1, 11))))
            cb.ax.yaxis.set_ticks(2*minorticks-1, minor=True)
        else:
            minorticks = np.concatenate((s.norm(np.linspace(0, 0.1, 11)), 
                                         s.norm(np.linspace(0.1, 1, 11))))
            cb.ax.yaxis.set_ticks(minorticks, minor=True)

    ax.set_aspect(1.0)
    
    scale = np.gcd.reduce([h, k, l])
    
    if (scale != 0):  h, k, l = np.array([h, k, l]) // scale

    H = str(h) if (h >= 0) else r'\bar{{{}}}'.format(h)
    K = str(k) if (k >= 0) else r'\bar{{{}}}'.format(k)
    L = str(l) if (l >= 0) else r'\bar{{{}}}'.format(l)
    
    d = np.round(d, int(-np.log10(tol)))
       
    ax.set_title(r'$({}{}{})\cdot[uvw]={}$'.format(H,K,L,d), fontsize='small')

    ax.minorticks_on()
    
    ax.set_aspect(aspect)
    
    uvw = np.array(['u','v','w'])
    
    var0 = np.repeat(uvw[np.argwhere(np.isclose(proj0,1))[0][0]],3)
    var1 = np.repeat(uvw[np.argwhere(np.isclose(proj1,1))[0][0]],3)
    
    coeff0 = np.round(proj0,4).astype(str)
    coeff1 = np.round(proj1,4).astype(str)
    
    for c in range(3):
        if (np.isclose(np.float(coeff0[c]),0)):
            coeff0[c] = '0'
            var0[c] = ''
        elif (np.isclose(np.float(coeff0[c]),1)):
            coeff0[c] = ''
        elif (np.isclose(np.float(coeff0[c]),-1)):
            coeff0[c] = '-'
            
        if (np.isclose(np.float(coeff1[c]),0)):
            coeff1[c] = '0'            
            var1[c] = ''            
        elif (np.isclose(np.float(coeff1[c]),1)):
            coeff1[c] = ''
        elif (np.isclose(np.float(coeff1[c]),-1)):
            coeff1[c] = '-'
            
    ax.set_xlabel(r'$['+coeff0[0]+var0[0]+','\
                       +coeff0[1]+var0[1]+','\
                       +coeff0[2]+var0[2]+']$')
                
    ax.set_ylabel(r'$['+coeff1[0]+var1[0]+','\
                       +coeff1[1]+var1[1]+','\
                       +coeff1[2]+var1[2]+']$')

    if (correlation == 'Correlation'):
        if (disorder == 'Moment'):
            label = r'$\langle\mathbf{S}(\mathbf{0})'\
                    r'\cdot\mathbf{S}(\mathbf{r})\rangle$'
        elif (disorder == 'Occupancy'):
            label = r'$\langle\sigma(\mathbf{0})'\
                    r'\cdot\sigma(\mathbf{r})\rangle$'
        else:
            label = r'$\langle\hat{\mathbf{u}}(\mathbf{0})'\
                    r'\cdot\hat{\mathbf{u}}(\mathbf{r})\rangle$'
    else:
        if (disorder == 'Moment'):
            label = r'$\langle|\mathbf{S}(\mathbf{0})'\
                    r'\cdot\mathbf{S}(\mathbf{r})|^2\rangle$'
        elif (disorder == 'Occupancy'):
            label = r'$\langle|\sigma(\mathbf{0})'\
                    r'\cdot\sigma(\mathbf{r})|^2\rangle$'
        else:
            label = r'$\langle|\hat{\mathbf{u}}(\mathbf{0})'\
                    r'\cdot\hat{\mathbf{u}}(\mathbf{r})|^2\rangle$'
                    
    cb.set_label(label)

    ax.axes.tick_params(labelsize='small')
    
    fig.tight_layout(pad=3.24)
    canvas.draw()
    
    return H, K, L, d

def plot_calc_3d(canvas, data, hkl, slice_hkl, i_hkl, T,
                 min_h, min_k, min_l, max_h, max_k, max_l, 
                 size_h, size_k, size_l, matrix_h, matrix_k, matrix_l, 
                 scale_h, scale_k, scale_l, norm, vmin, vmax):
        
    extents_h = [min_k, max_k, min_l, max_l]
    extents_k = [min_h, max_h, min_l, max_l]
    extents_l = [min_h, max_h, min_k, max_k]
                
    ext_min_h, ext_max_h = __transform_extents(matrix_h, extents_h)
    ext_min_k, ext_max_k = __transform_extents(matrix_k, extents_k)
    ext_min_l, ext_max_l = __transform_extents(matrix_l, extents_l)
    
    extents_h = __extents(min_k, min_l, max_k, max_l, size_k, size_l)
    extents_k = __extents(min_h, min_l, max_h, max_l, size_h, size_l)
    extents_l = __extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_h = __offset(matrix_h, min_l)
    offset_k = __offset(matrix_k, min_l)
    offset_l = __offset(matrix_l, min_k)
    
    aligned = np.allclose(T, np.eye(3))
        
    if (norm == 'Logarithmic'):
        if (np.isclose(vmin, 0) and np.isclose(vmax, 0)):
            vmin, vmax = 1e-3, 1e-2
        normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
                
    fig = canvas.figure
    fig.clear()   
    
    ax = fig.add_subplot(111)
    
    ax.set_aspect(1.)
    
    if (hkl == 'h ='):
    
        im = ax.imshow(data[i_hkl,:,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_h)
        
        ax.set_title(r'$h={}$'.format(slice_hkl), fontsize='small') 
        
        if aligned:
            ax.set_xlabel(r'$(0k0)$', fontsize='small')
            ax.set_ylabel(r'$(00l)$', fontsize='small')
        else:
            ax.set_xlabel(r'$(\bar{k}k0)$', fontsize='small')
            ax.set_ylabel(r'$(00l)$', fontsize='small')

        matrix, offset, scale = matrix_h, offset_h, scale_h
        ext_min, ext_max = ext_min_h, ext_max_h
        
    elif (hkl == 'k ='):
        
        im = ax.imshow(data[:,i_hkl,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_k)
    
        ax.set_title(r'$k={}$'.format(slice_hkl), fontsize='small')
   
        if aligned:
            ax.set_xlabel(r'$(h00)$', fontsize='small')
            ax.set_ylabel(r'$(00l)$', fontsize='small')
        else:
            ax.set_xlabel(r'$(hh0)$', fontsize='small')
            ax.set_ylabel(r'$(00l)$', fontsize='small')
        
        matrix, offset, scale = matrix_k, offset_k, scale_k
        ext_min, ext_max = ext_min_k, ext_max_k
        
    else:
    
        im = ax.imshow(data[:,:,i_hkl].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_l)
        
        ax.set_title(r'$l={}$'.format(slice_hkl), fontsize='small') 
        
        if aligned:
            ax.set_xlabel(r'$(h00)$', fontsize='small') 
            ax.set_ylabel(r'$(0k0)$', fontsize='small') 
        else:
            ax.set_xlabel(r'$(hh0)$', fontsize='small')
            ax.set_ylabel(r'$(\bar{k}k0)$', fontsize='small')
        
        matrix, offset, scale = matrix_l, offset_l, scale_l
        ext_min, ext_max = ext_min_l, ext_max_l
   
    trans = mtransforms.Affine2D()
    
    trans.set_matrix(matrix)
            
    shift = mtransforms.Affine2D().translate(offset,0)
    
    ax.set_aspect(scale)
    
    trans_data = trans+shift+ax.transData
    
    im.set_transform(trans_data)
    
    ax.set_xlim(ext_min[0]+offset,ext_max[0]+offset)
    ax.set_ylim(ext_min[1],ext_max[1])   

    ax.xaxis.tick_bottom()
    
    ax.minorticks_on()
    
    ax.axes.tick_params(labelsize='small')

    fig.tight_layout(pad=3.24)
    
    cb = fig.colorbar(im, ax=ax)
    cb.ax.minorticks_on()
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
    else:
        cb.ax.xaxis.set_major_locator(plt.NullLocator())
        cb.ax.xaxis.set_minor_locator(plt.NullLocator())
        
    cb.ax.tick_params(labelsize='small') 
    canvas.draw()