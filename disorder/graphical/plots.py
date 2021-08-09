#!/usr/bin/env/python3

import re
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

from matplotlib import ticker
from matplotlib.ticker import Locator

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
matplotlib.rcParams['mathtext.cal'] = 'sans'
matplotlib.rcParams['mathtext.rm'] = 'sans'
matplotlib.rcParams['mathtext.sf'] = 'sans'
matplotlib.rcParams['mathtext.tt'] = 'monospace'

def _extents(min_x, min_y, max_x, max_y, size_x, size_y):
        
    dx = 0 if size_x <= 1 else (max_x-min_x)/(size_x-1)
    dy = 0 if size_y <= 1 else (max_y-min_y)/(size_y-1)
            
    return [min_x-dx/2, max_x+dx/2, min_y-dy/2, max_y+dy/2]
        
def _transform_extents(matrix, extents):

    ext_min = np.dot(matrix[0:2,0:2], extents[0::2])
    ext_max = np.dot(matrix[0:2,0:2], extents[1::2])
    
    return ext_min, ext_max

def _offset(matrix, minimum):
    
    return -np.dot(matrix, [0,minimum,0])[0]

def plot_exp_h(canvas, data, h, ih, min_k, min_l, max_k, max_l, size_k, size_l, 
               matrix_h, scale_h, norm, vmin, vmax):
        
    extents_h = [min_k, max_k, min_l, max_l]
    
    ext_min_h, ext_max_h = _transform_extents(matrix_h, extents_h)
    
    extents_h = _extents(min_k, min_l, max_k, max_l, size_k, size_l)
    
    offset_h = _offset(matrix_h, min_l)

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
    
    ext_min_k, ext_max_k = _transform_extents(matrix_k, extents_k)
    
    extents_k = _extents(min_h, min_l, max_h, max_l, size_h, size_l)
    
    offset_k = _offset(matrix_k, min_l)

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
    
    ext_min_l, ext_max_l = _transform_extents(matrix_l, extents_l)
    
    extents_l = _extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_l = _offset(matrix_l, min_k)
    
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
    
    ext_min_h, ext_max_h = _transform_extents(matrix_h, extents_h)
    ext_min_k, ext_max_k = _transform_extents(matrix_k, extents_k)
    ext_min_l, ext_max_l = _transform_extents(matrix_l, extents_l)
    
    extents_h = _extents(min_k, min_l, max_k, max_l, size_k, size_l)
    extents_k = _extents(min_h, min_l, max_h, max_l, size_h, size_l)
    extents_l = _extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_h = _offset(matrix_h, min_l)
    offset_k = _offset(matrix_k, min_l)
    offset_l = _offset(matrix_l, min_k)
    
    if (norm == 'Logarithmic'):
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

def chi_sq(canvas, plot0, plot1, acc_moves, rej_moves,
           temperature, energy, chi_sq, scale):

    fig = canvas.figure
    fig.clear()   
    
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    
    if (plot0 == 'Accepted'):
        line0, = ax0.semilogy(acc_moves, 'C0')
        ax0.set_ylabel(r'Accepted $\chi^2$', fontsize='small')    
    elif (plot0 == 'Rejected'):
        line0, = ax0.semilogy(rej_moves, 'C0')
        ax0.set_ylabel(r'Rejected $\chi^2$', fontsize='small') 
    elif (plot0 == 'Temperature'):              
        line0, = ax0.semilogy(temperature, 'C0')
        ax0.set_ylabel(r'Temperatrue $T$', fontsize='small') 
    elif (plot0 == 'Energy'):              
        line0, = ax0.plot(energy, 'C0')
        ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        ax0.set_ylabel(r'Energy $\Delta\chi^2$', fontsize='small') 
    elif (plot0 == 'Chi-squared'):              
        line0, = ax0.semilogy(chi_sq, 'C0')
        ax0.set_ylabel(r'$\chi^2$', fontsize='small') 
    else:
        line0, = ax0.plot(scale, 'C0')
        ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        
        ax0.set_ylabel(r'Scale factor', fontsize='small') 

    if (plot1 == 'Accepted'):
        line1, = ax1.semilogy(acc_moves, 'C1')
        ax1.set_ylabel(r'Accepted $\chi^2$', fontsize='small')    
    elif (plot1 == 'Rejected'):
        line1, = ax1.semilogy(rej_moves, 'C1')
        ax1.set_ylabel(r'Rejected $\chi^2$', fontsize='small') 
    elif (plot1 == 'Temperature'):              
        line1, = ax1.semilogy(temperature, 'C1')
        ax1.set_ylabel(r'Temperature $T$', fontsize='small') 
    elif (plot1 == 'Energy'):              
        line1, = ax1.plot(energy, 'C1')
        ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        ax1.set_ylabel(r'Energy $\Delta\chi^2$', fontsize='small') 
    elif (plot1 == 'Chi-squared'):              
        line1, = ax1.semilogy(chi_sq, 'C1')
        ax1.set_ylabel(r'$\chi^2$', fontsize='small') 
    else:
        line1, = ax1.plot(scale, 'C1')
        ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        ax1.set_ylabel(r'Scale factor', fontsize='small') 
     
    ax0.set_xlabel(r'Moves', fontsize='small')
    ax1.set_xlabel(r'Moves', fontsize='small')   
     
    ax0.axes.tick_params(labelsize='small')
    ax1.axes.tick_params(labelsize='small')
           
    ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    
    ax0.minorticks_on()
    ax1.minorticks_on()
    
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #ax.set_title(r'\chi^2$', fontsize='small') 
    
    fig.tight_layout(pad=3.24)
    canvas.draw()
    
    return ax0, ax1, line0, line1
    
def fast_chi_sq(canvas, ax0, ax1, line0, line1, plot0, plot1, data0, data1):
    
    line0.set_data(np.arange(len(data0)), data0)
    line1.set_data(np.arange(len(data1)), data1)

    ax0.relim()
    ax1.relim()
    
    ax0.autoscale_view()
    ax1.autoscale_view()
    
    canvas.draw_idle() 
    
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
    
def correlations_1d(canvas, d, data, error, atm_pair, disorder, 
                    correlation, average, norm, atoms, pairs):
        
    fig = canvas.figure
    fig.clear()   
        
    ax = fig.add_subplot(111)

    if (correlation == 'Correlation'):   
        ax.axhline(y=0, xmin=0, xmax=1, color='k', linestyle='-', linewidth=1)
        
    if average:  
        
        ax.scatter(d, data, marker='o', clip_on=False, zorder=50)
        
    else:
        
        for atom, pair in zip(atoms, pairs):
            
            if (atom == r'self-correlation'):
                mask = atm_pair == '0'
                label = atom
                
            else:
                mask = atm_pair == atom+'_'+pair
                
                atom0 = re.sub(r'[^a-zA-Z]', '', atom)
                atom1 = re.sub(r'[^a-zA-Z]', '', pair)
                
                pre0, post0 = atom.split(atom0)
                pre1, post1 = pair.split(atom1)
                
                label = r'$^{{{}}}${}$^{{{}}}$-'\
                        r'$^{{{}}}${}$^{{{}}}$'.format(pre0, atom0, post0,
                                                       pre1, atom1, post1)
                        
            ax.errorbar(d[mask], data[mask], yerr=1.96*np.sqrt(error[mask]),
                        fmt='o', ls='none', clip_on=False, zorder=50, 
                        label=label)
                            
        if (len(atoms) > 0):
            ax.legend(loc='best', frameon=True, 
                      fancybox=True, fontsize='small')        
          
    ax.minorticks_on()
   
    if (norm == 'Logarithmic'):
        ax.set_yscale('symlog', linthresh=0.1, linscale=(1-10**-1))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(MinorSymLogLocator(1e-1))
    
    x1,x2,y1,y2 = ax.axis()
    ax.set_xlim([0,x2])

    if (correlation == 'Correlation'):                
        ax.set_ylim([-1,1])
    else:
        ax.set_ylim([0,1])
        
    ax.set_xlabel(r'$r$ [Å]')
    
    if (correlation == 'Correlation'):
        if (disorder == 'Moment'):
            label = r'$\langle\mathbf{S}(0)\cdot\mathbf{S}(r)\rangle$'
        elif (disorder == 'Occupancy'):
            label = r'$\langle\sigma(0)\cdot\sigma(r)\rangle$'
        else:
            label = r'$\langle\hat{\mathbf{u}}(0)'\
                    r'\cdot\hat{\mathbf{u}}(r)\rangle$'
    else:
        if (disorder == 'Moment'):
            label = r'$\langle|\mathbf{S}(0)\cdot\mathbf{S}(r)|^2\rangle$'
        elif (disorder == 'Occupancy'):
            label = r'$\langle|\sigma(0)\cdot\sigma(r)|^2\rangle$'
        else:
            label = r'$\langle|\hat{\mathbf{u}}(0)'\
                    r'\cdot\hat{\mathbf{u}}(r)|^2\rangle$'            

    ax.set_ylabel(label)
    ax.axes.tick_params(labelsize='small')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout(pad=3.24)
    canvas.draw()
    
def _mask_plane(dx, dy, dz, h, k, l, d, A, B, tol):
             
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
         
         proj0 = np.array([pu,pv,pw])
         proj1 = np.array([qu,qv,qw])
                                 
         scale_d0 = proj0.max()
         scale_d1 = proj1.max()
         
         proj0 = proj0/scale_d0
         proj1 = proj1/scale_d1
         
         cor_aspect = scale_d0/scale_d1
       
         d0 = (px*dx[plane]+py*dy[plane]+pz*dz[plane])*scale_d0
         d1 = (qx*dx[plane]+qy*dy[plane]+qz*dz[plane])*scale_d1
         
         return cor_aspect, proj0, proj1, d0, d1, plane
    
def correlations_3d(canvas, dx, dy, dz, h, k, l, d, data, error, atm_pair3d, 
                    disorder, correlation, average, norm, atoms, pairs, 
                    A, B, tol):
    
    aspect, proj0, proj1, d0, d1, plane = _mask_plane(dx, dy, dz, h, k, l, 
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

def plot_calc(canvas, data, hkl, slice_hkl, i_hkl, T,
              min_h, min_k, min_l, max_h, max_k, max_l, size_h, size_k, size_l, 
              matrix_h, matrix_k, matrix_l, scale_h, scale_k, scale_l,
              norm, vmin, vmax):
        
    extents_h = [min_k, max_k, min_l, max_l]
    extents_k = [min_h, max_h, min_l, max_l]
    extents_l = [min_h, max_h, min_k, max_k]
                
    ext_min_h, ext_max_h = _transform_extents(matrix_h, extents_h)
    ext_min_k, ext_max_k = _transform_extents(matrix_k, extents_k)
    ext_min_l, ext_max_l = _transform_extents(matrix_l, extents_l)
    
    extents_h = _extents(min_k, min_l, max_k, max_l, size_k, size_l)
    extents_k = _extents(min_h, min_l, max_h, max_l, size_h, size_l)
    extents_l = _extents(min_h, min_k, max_h, max_k, size_h, size_k)
    
    offset_h = _offset(matrix_h, min_l)
    offset_k = _offset(matrix_k, min_l)
    offset_l = _offset(matrix_l, min_k)
    
    aligned = np.allclose(T, np.eye(3))
        
    if (norm == 'Logarithmic'):
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