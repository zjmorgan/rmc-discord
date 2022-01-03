#!/usr/bin/env/python3

import numpy as np

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

from matplotlib import ticker
from matplotlib.ticker import Locator
from matplotlib.patches import Polygon

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
        
    def get_aspect(self):
        
        width, height = self.ax.get_figure().get_size_inches()
        _, _, w, h = self.ax.get_position().bounds
        
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
    
        disp_ratio = (height*h)/(width*w)
        data_ratio = (ymax-ymin)/(xmax-xmin)
            
        return disp_ratio/data_ratio

    def save_figure(self, filename):

        self.fig.savefig(filename, bbox_inches='tight', transparent=False)

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

    def set_aspect(self, value):

        self.ax.set_aspect(value)

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

    def set_limits(self, vmin=None, vmax=None, twin=False):

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

        min_x = extents[0]-dx/2
        max_x = extents[1]+dx/2

        min_y = extents[2]-dy/2
        max_y = extents[3]+dy/2

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

        self.im.set_norm(self.norm)

        if self.im.colorbar is not None:

            orientation = self.im.colorbar.orientation

            self.remove_colorbar()
            self.create_colorbar(orientation, norm)

            if (norm.lower() == 'symlog'):
                self.cb.locator = ticker.SymmetricalLogLocator(linthresh=0.1,
                                                               base=10)
                self.cb.update_ticks()

    def update_normalization(self, norm='linear'):

        if self.im is not None:

            vmin, vmax = self.im.get_clim()

            self.set_normalization(vmin, vmax, norm)

    def reformat_colorbar(self, formatstr='{:.1f}'):

        if (self.cb.orientation == 'vertical'):
            ticks = self.cb.ax.get_yticks()
        else:
            ticks = self.cb.ax.get_xticks()

        vmin, vmax = self.cb.vmin, self.cb.vmax

        inv = self.norm.inverse
        tscale = inv((ticks-vmin)/(vmax-vmin))
        labels = [formatstr.format(t) for t in tscale]

        norm = self.norm
        minorticks = []

        if (vmin < ticks[0]):
            vn = 11 if vmin >= -0.1 and vmin <= 0.1 else 10
            tmin = inv((np.array([ticks[0]])-vmin)/(vmax-vmin))[0]
            if (vmin >= -0.1 and vmin <= 0.1):
                tn = int(vmin/-0.01)
                nmin = -0.1 if vmin < 0.0 else 0.0
            else:
                tn = int(vmin/tmin)
                nmin = tmin*10

            values = (vmax-vmin)*norm(np.linspace(nmin, tmin, vn))[-tn:-1]+vmin
            minorticks += values.tolist()

        for i in range(len(ticks)-1):
            tmin = inv((np.array([ticks[i]])-vmin)/(vmax-vmin))[0]
            tmax = inv((np.array([ticks[i+1]])-vmin)/(vmax-vmin))[0]

            tn = 11 if tmin >= -0.1 and tmax <= 0.1 else 10

            values = (vmax-vmin)*norm(np.linspace(tmin, tmax, tn))[1:-1]+vmin
            minorticks += values.tolist()

        if (vmax > ticks[-1]):
            vn = 11 if vmax >= -0.1 and vmax <= 0.1 else 10
            tmax = inv((np.array([ticks[-1]])-vmin)/(vmax-vmin))[0]
            if (vmax >= -0.1 and vmax <= 0.1):
                tn = int(vmax/0.01)
                nmax = 0.1 if vmax > 0.0 else 0.0
            else:
                tn = int(vmax/tmax)
                nmax =  tmax*10

            values = (vmax-vmin)*norm(np.linspace(tmax, nmax, vn))[1:tn]+vmin
            minorticks += values.tolist()

        if (self.cb.orientation == 'vertical'):
            self.cb.ax.set_yticklabels(labels)
            self.cb.ax.yaxis.set_ticks(minorticks, minor=True)
        else:
            self.cb.ax.set_xticklabels(labels)
            self.cb.ax.xaxis.set_ticks(minorticks, minor=True)

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

    def remove_colorbar(self):

        if self.im.colorbar is not None:

            self.im.colorbar.remove()
            
    def set_colorbar_label(self, label):

        if self.im.colorbar is not None:

            self.im.colorbar.set_label(label)

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
        
        self.transformation = transformation
        self.offset = offset

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
        
        for p in reversed(self.ax.patches):
            p.remove()
        
        x = [extents[1],ext_max[0]+offset,ext_max[0]+offset]
        y = [ext_min[1],ext_min[1],ext_max[1]]
        p = Polygon(np.column_stack((x,y)), fc='w', ec='w')
        self.ax.add_patch(p)
        
        x = [extents[0],extents[0],extents[0]+offset*2]
        y = [ext_min[1],ext_max[1],ext_max[1]]
        p = Polygon(np.column_stack((x,y)), fc='w', ec='w')
        self.ax.add_patch(p)
        
    def draw_line(self, xlim=None, ylim=None):
                        
        if xlim is None: xlim = self.ax.get_xlim()
        if ylim is None: ylim = self.ax.get_ylim()
        
        self.ax.plot(xlim, ylim, color='w')
        
    def add_text(self, x, y, s, color='w'):
        
        self.ax.text(x, y, s, color=color, ha='center', va='center')

class Scatter(Plot):

    def __init__(self, canvas):

        super(Scatter, self).__init__(canvas)

        self.s = None
        self.__color_limits()

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

        self.s.set_norm(self.norm)

        if self.s.colorbar is not None:

            orientation = self.s.colorbar.orientation

            self.remove_colorbar()
            self.create_colorbar(orientation, norm)

            if (norm.lower() == 'symlog'):
                self.cb.locator = ticker.SymmetricalLogLocator(linthresh=0.1,
                                                               base=10)
                self.cb.update_ticks()

    def update_normalization(self, norm='linear'):

        if self.s is not None:

            vmin, vmax = self.s.get_clim()

            self.set_normalization(vmin, vmax, norm)

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

    def remove_colorbar(self):

        if self.s.colorbar is not None:

            self.s.colorbar.remove()

    def set_colorbar_label(self, label):

        if self.s.colorbar is not None:

            self.s.colorbar.set_label(label)

    def reset_colorbar_limits(self):

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

    def reformat_colorbar(self, formatstr='{:.1f}'):

        if (self.cb.orientation == 'vertical'):
            ticks = self.cb.ax.get_yticks()
        else:
            ticks = self.cb.ax.get_xticks()

        vmin, vmax = self.cb.vmin, self.cb.vmax

        inv = self.norm.inverse
        tscale = inv((ticks-vmin)/(vmax-vmin))
        labels = [formatstr.format(t) for t in tscale]

        norm = self.norm
        minorticks = []

        if (vmin < ticks[0]):
            vn = 11 if vmin >= -0.1 and vmin <= 0.1 else 10
            tmin = inv((np.array([ticks[0]])-vmin)/(vmax-vmin))[0]
            if (vmin >= -0.1 and vmin <= 0.1):
                tn = int(vmin/-0.01)
                nmin = -0.1 if vmin < 0.0 else 0.0
            else:
                tn = int(vmin/tmin)
                nmin = tmin*10

            values = (vmax-vmin)*norm(np.linspace(nmin, tmin, vn))[-tn:-1]+vmin
            minorticks += values.tolist()

        for i in range(len(ticks)-1):
            tmin = inv((np.array([ticks[i]])-vmin)/(vmax-vmin))[0]
            tmax = inv((np.array([ticks[i+1]])-vmin)/(vmax-vmin))[0]

            tn = 11 if tmin >= -0.1 and tmax <= 0.1 else 10

            values = (vmax-vmin)*norm(np.linspace(tmin, tmax, tn))[1:-1]+vmin
            minorticks += values.tolist()

        if (vmax > ticks[-1]):
            vn = 11 if vmax >= -0.1 and vmax <= 0.1 else 10
            tmax = inv((np.array([ticks[-1]])-vmin)/(vmax-vmin))[0]
            if (vmax >= -0.1 and vmax <= 0.1):
                tn = int(vmax/0.01)
                nmax = 0.1 if vmax > 0.0 else 0.0
            else:
                tn = int(vmax/tmax)
                nmax =  tmax*10

            values = (vmax-vmin)*norm(np.linspace(tmax, nmax, vn))[1:tn]+vmin
            minorticks += values.tolist()

        if (self.cb.orientation == 'vertical'):
            self.cb.ax.set_yticklabels(labels)
            self.cb.ax.yaxis.set_ticks(minorticks, minor=True)
        else:
            self.cb.ax.set_xticklabels(labels)
            self.cb.ax.xaxis.set_ticks(minorticks, minor=True)