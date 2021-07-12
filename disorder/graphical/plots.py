#!/usr/bin/env/python3

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

def _extents(min_x, min_y, max_x, max_y, size_x, size_y):
        
    dx = (max_x-min_x)/(size_x-1)
    dy = (max_y-min_y)/(size_y-1)
            
    return [min_x-dx/2, max_x+dx/2, min_y-dy/2, max_y+dy/2]
        
def _transform_extents(matrix, extents):

    ext_min = np.dot(matrix[0:2,0:2], extents[0::2])
    ext_max = np.dot(matrix[0:2,0:2], extents[1::2])
    
    return ext_min, ext_max

def _offset(matrix, minimum):
    return -np.dot(matrix, [0,minimum,0])[0]

def plot_exp(canvas, data, 
             h, k, l, 
             ih, ik, il, 
             min_h, min_k, min_l, 
             max_h, max_k, max_l, 
             size_h, size_k, size_l, 
             matrix_h, matrix_k, matrix_l,
             scale_h, scale_k, scale_l,
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
    
    ax_h = fig.add_subplot(131)
    ax_k = fig.add_subplot(132)
    ax_l = fig.add_subplot(133)
    
    im_h = ax_h.imshow(data[ih,:,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_h,
                       zorder=100)
    
    im_k = ax_k.imshow(data[:,ik,:].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_k,
                       zorder=100)
    
    im_l = ax_l.imshow(data[:,:,il].T,
                       norm=normalize,
                       interpolation='nearest', 
                       origin='lower',
                       extent=extents_l,
                       zorder=100)
    
    trans_h = mtransforms.Affine2D()
    trans_k = mtransforms.Affine2D()
    trans_l = mtransforms.Affine2D()
    
    trans_h.set_matrix(matrix_h)
    trans_k.set_matrix(matrix_k)
    trans_l.set_matrix(matrix_l)
    
    shift_h = mtransforms.Affine2D().translate(offset_h,0)
    shift_k = mtransforms.Affine2D().translate(offset_k,0)
    shift_l = mtransforms.Affine2D().translate(offset_l,0)
    
    ax_h.set_aspect(scale_h)
    ax_k.set_aspect(scale_k)
    ax_l.set_aspect(scale_l)
    
    trans_data_h = trans_h+shift_h+ax_h.transData
    trans_data_k = trans_k+shift_k+ax_k.transData
    trans_data_l = trans_l+shift_l+ax_l.transData
    
    im_h.set_transform(trans_data_h)
    im_k.set_transform(trans_data_k)
    im_l.set_transform(trans_data_l)
    
    ax_h.set_xlim(ext_min_h[0]+offset_h,ext_max_h[0]+offset_h)
    ax_h.set_ylim(ext_min_h[1],ext_max_h[1])
    
    ax_k.set_xlim(ext_min_k[0]+offset_k,ext_max_k[0]+offset_k)
    ax_k.set_ylim(ext_min_k[1],ext_max_k[1])
    
    ax_l.set_xlim(ext_min_l[0]+offset_l,ext_max_l[0]+offset_l)
    ax_l.set_ylim(ext_min_l[1],ext_max_l[1])
            
    ax_h.xaxis.tick_bottom()
    ax_k.xaxis.tick_bottom()
    ax_l.xaxis.tick_bottom()
    
    ax_h.axes.tick_params(labelsize='small')
    ax_k.axes.tick_params(labelsize='small')
    ax_l.axes.tick_params(labelsize='small')
    
    ax_h.set_title(r'$h='+str(h)+'$', fontsize='small') 
    ax_k.set_title(r'$k='+str(k)+'$', fontsize='small') 
    ax_l.set_title(r'$l='+str(l)+'$', fontsize='small') 
    
    ax_h.set_xlabel(r'$k$', fontsize='small')
    ax_h.set_ylabel(r'$l$', fontsize='small')
    
    ax_k.set_xlabel(r'$h$', fontsize='small')
    ax_k.set_ylabel(r'$l$', fontsize='small')
    
    ax_l.set_xlabel(r'$h$', fontsize='small') 
    ax_l.set_ylabel(r'$k$', fontsize='small') 
    
    ax_h.minorticks_on()
    ax_k.minorticks_on()
    ax_l.minorticks_on()
    
    fig.tight_layout(pad=3.24)
 
    cb = fig.colorbar(im_l, ax=[ax_h,ax_k,ax_l])
    cb.ax.minorticks_on()           
    
    if (norm == 'Linear'):
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
    else:
        cb.ax.xaxis.set_major_locator(plt.NullLocator())
        cb.ax.xaxis.set_minor_locator(plt.NullLocator())

    cb.ax.tick_params(labelsize='small') 
    canvas.draw()