#!/usr/bin/env/python3

import numpy as np

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms

from matplotlib import ticker
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

class Plot():
    """
    General plot.

    Parameters
    ----------
    canvas : canvas
        Canvas for embedding in GUI.

    Attributes
    ----------
    canvas : canvas
        Qt canvas.
    fig : figure
        Canvas figure.
    ax : axis
        Figure axis.

    Methods
    -------
    get_aspect()
        Apect ratio of figure.
    set_aspect()
        Update apect ratio of figure.
    set_labels()
        Update axis titles and labels.
    clear_canvas()
        Clear canvas and remove axis.
    draw_canvas()
        Draw canvas.
    tight_layout()
        Use a tight layout for figure.
    save_figure()
        Save figure to file.

    """

    def __init__(self, canvas):

        self.canvas = canvas
        self.fig = canvas.figure
        self.ax = canvas.figure.add_subplot(111)
        self.ax.minorticks_on()

    def get_aspect(self):
        """
        Apect ratio of figure.

        Returns
        -------
        value : float
            Ratio of height to width.

        """

        width, height = self.ax.get_figure().get_size_inches()
        _, _, w, h = self.ax.get_position().bounds

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        disp_ratio = (height*h)/(width*w)
        data_ratio = (ymax-ymin)/(xmax-xmin)

        return disp_ratio/data_ratio

    def set_aspect(self, value):
        """
        Update apect ratio of figure.

        Parameters
        ----------
        value : float
            Ratio of height to width.

        """

        self.ax.set_aspect(value)

    def set_labels(self, title='', xlabel='', ylabel=''):
        """
        Update axis titles and labels.

        Parameters
        ----------
        title : str
            Title of the figure. Default is ``''``.
        xlabel : str
            Label for x-axis. Default is ``''``.
        ylabel : str
            Label for y-axis. Default is ``''``.

        """

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def clear_canvas(self):
        """
        Clear canvas and remove axis.

        """

        self.ax.remove()
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.minorticks_on()

    def draw_canvas(self):
        """
        Draw canvas.

        """

        self.canvas.draw_idle()

    def tight_layout(self, pad=3.24):
        """
        Use a tight layout for figure.

        Parameters
        ----------
        pad : float, optional
            Pad size. Default is ``3.24``.

        """

        self.fig.tight_layout(pad=pad)

    def save_figure(self, filename):
        """
        Save figure to file.

        Parameters
        ----------
        filename : str
            Name of file. Allowed extensions are ``*.pdf`` and ``*.png``.

        """

        self.fig.savefig(filename, bbox_inches='tight', transparent=False)

    def show(self):
        """
        Show figure.

        """

        self.fig.show()

class Line(Plot):
    """
    Line plot.

    Parameters
    ----------
    canvas : canvas
        Canvas for embedding in GUI.

    Attributes
    ----------
    pl : list
        Line plots.
    colors : color cycler
        Cycle of colors for plotting.

    Methods
    -------
    set_labels()
        Update axis titles and labels.
    clear_canvas()
        Clear canvas and remove axis.
    clear_lines()
        Clear line plots.
    set_normalization()
        Update data normalization.
    set_limits()
        Update data limits.
    reset_view()
        Autoscale data.
    update_data()
        Replace indexed data.
    get_data()
        Indexed data.
    plot_data()
        Plot data sequentially.
    show_legend()
        Display legend.
    draw_horizontal()
        Draw horizontal line.
    use_scientific()
        Use scientific notation.

    """

    def __init__(self, canvas):

        super(Line, self).__init__(canvas)

        self.pl = []

        self.__hl = None
        self.__twin_ax = None

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        prop_cycle = matplotlib.rcParams['axes.prop_cycle']
        self.colors = cycle(prop_cycle.by_key()['color'])

    def __get_axis(self, twin=False):

        if not twin:
            return self.ax
        else:
            if self.__twin_ax is None:
                self.__twin_ax = self.ax.twinx()
            return self.__twin_ax

    def set_labels(self, title='', xlabel='', ylabel='', twin_ylabel=''):
        """
        Update axis titles and labels.

        Parameters
        ----------
        title : str
            Title of the figure. Default is ``''``.
        xlabel : str
            Label for x-axis. Default is ``''``.
        ylabel : str
            Label for y-axis. Default is ``''``.
        twin_ylabel : str
            Label for twin y-axis. Default is ``''``.

        """

        super().set_labels(title=title, xlabel=xlabel, ylabel=ylabel)

        if self.__twin_ax:
            self.__twin_ax.set_ylabel(twin_ylabel)

    def clear_canvas(self):
        """
        Clear canvas and remove axis.

        """

        super().clear_canvas()
        self.clear_lines()

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

    def clear_lines(self):
        """
        Clear line plots.

        """

        if self.pl:
            for p in self.pl:
                p.lines[0].remove()
            self.pl = []

        if self.__hl:
            self.__hl.remove()
            self.__hl = None
        if self.__twin_ax:
            self.__twin_ax.remove()
            self.__twin_ax = None

        if self.ax.get_legend():
            self.ax.get_legend().remove()
        self.set_labels()

        prop_cycle = matplotlib.rcParams['axes.prop_cycle']
        self.colors = cycle(prop_cycle.by_key()['color'])

    def set_normalization(self, norm='linear', twin=False):
        """
        Update data normalization.

        Parameters
        ----------
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.
        twin : bool, optional
            Apply to twin axis. The default is ``False``.

        """

        ax = self.__get_axis(twin)

        if (norm.lower() == 'linear'):
            ax.set_yscale('linear')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        elif (norm.lower() == 'logarithmic'):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.yaxis.set_minor_locator(ticker.LogLocator())
        else:
            thresh, scale = 0.1, 0.9
            ax.set_yscale('symlog', linthresh=thresh, linscale=scale)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            sym_log = ticker.SymmetricalLogLocator(linthresh=0.1, base=10,
                                                   subs=np.linspace(0.1,0.9,9))
            ax.yaxis.set_minor_locator(sym_log)

    def set_limits(self, vmin=None, vmax=None, twin=False):
        """
        Update data limits.

        Parameters
        ----------
        vmin : float, optional
            Minimum ordinate value. The default is ``None``.
        vmax : float, optional
            Maximum ordinate value. The default is ``None``.
        twin : bool, optional
            Apply to twin axis. The default is ``False``.

        """

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
        """
        Autoscale data. Limits and scale are reset.

        Parameters
        ----------
        twin : bool, optional
            Apply to twin axis. The default is ``False``.

        """

        ax = self.__get_axis(twin)

        ax.relim()
        ax.autoscale_view()

    def update_data(self, x, y, i=0):
        """
        Replace indexed data on existing plot.

        Parameters
        ----------
        x : 1d array or list
            Data correpsonding to abscissa.
        y : 1d array or list
            Data correpsonding to ordinate.
        i : int, optional
            Index of dataset. The default is ``0``.

        """

        self.pl[i].lines[0].set_data(x, y)

    def get_data(self, i=0):
        """
        Indexed data.

        Parameters
        ----------
        i : int, optional
            Index of dataset. The default is ``0``.

        Returns
        -------
        x : 1d array
            Data correpsonding to abscissa.
        y : 1d array
            Data correpsonding to ordinate.

        """

        return self.pl[i].lines[0].get_xydata().T

    def plot_data(self, x, y, yerr=None, marker='o', label='', twin=False):
        """
        Plot data sequentially.

        Parameters
        ----------
        x : 1d array or list
            Data correpsonding to abscissa.
        y : 1d array or list
            Data correpsonding to ordinate.
        yerr : 1d array or list, optional
            Error correpsonding to ordinate. The default is ``None``.
        marker : str, optional
            Marker symbol. The default is ``'o'``.
        label : str, optional
            Label for legend. The default is ``''``.
        twin : bool, optional
            Plot on twin axis. The default is ``False``.

        """

        ax = self.__get_axis(twin)

        color = next(self.colors)

        err = ax.errorbar(x, y, yerr=yerr, fmt=marker, label=label,
                          color=color, ecolor=color, clip_on=False)

        self.pl.append(err)

    def show_legend(self):
        """
        Display legend.

        """

        handles = [p for p in self.pl if p.get_label() != '']
        labels = [p.get_label() for p in handles]
        self.ax.legend(handles, labels)

    def draw_horizontal(self):
        """
        Draw horizontal line at zero ordinate.

        """

        self.__hl = self.ax.axhline(y=0, xmin=0, xmax=1, color='k', \
                                    linestyle='-', linewidth=0.8)

    def use_scientific(self, twin=False):
        """
        Use scientific notation.

        Parameters
        ----------
        twin : bool, optional
            Apply to twin axis. The default is ``False``.

        """

        ax = self.__get_axis(twin)

        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

class HeatMap(Plot):
    """
    Intensity heat map plot.

    Parameters
    ----------
    canvas : canvas
        Canvas for embedding in GUI.

    Attributes
    ----------
    im : image
        Image plot.
    norm : normalization
        Color normalization.
    cmap : colormap
        Colormap.

    Methods
    -------
    set_normalization()
        Update data normalization.
    update_normalization()
        Replace normalization.
    update_colormap()
        Replace colormap.
    create_colorbar()
        Create colorbar.
    remove_colorbar()
        Remove colorbar.
    set_colorbar_label()
        Update colorbar label.
    reset_color_limits()
        Autoscale color limits.
    update_data()
        Replace data.
    get_data()
        Data.
    plot_data()
        Plot data.
    transform_axes()
        Skew axes according to a transformation matrix.
    draw_line()
        Draw line with specified extents.
    add_text()
        Add text at specified location.

    """

    def __init__(self, canvas):

        super(HeatMap, self).__init__(canvas)

        self.im = None
        self.norm = None
        self.cmap = None

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
        """
        Update data normalization.

        Parameters
        ----------
        vmin : float
            Minimum data value.
        vmax : float
            Maximum data value.
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

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
                sym_log = ticker.SymmetricalLogLocator(linthresh=0.1, base=10)
                self.cb.locator = sym_log
                subs = np.linspace(0.1,0.9,9)
                sym_log = ticker.SymmetricalLogLocator(linthresh=0.1, base=10,
                                                       subs=subs)
                self.cb.minorlocator = sym_log
                self.cb.formatter = ticker.ScalarFormatter()
                self.cb.update_ticks()

    def update_normalization(self, norm='linear'):
        """
        Replace normalization.

        Parameters
        ----------
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

        if self.im is not None:

            vmin, vmax = self.im.get_clim()

            self.set_normalization(vmin, vmax, norm)

    def update_colormap(self, category='sequential'):
        """
        Replace colormap.

        Parameters
        ----------
        category : str, optional
            Update colormap. Options are ``'sequential'``, ``'diverging'``, or
            ``'binary'``. The default is ``'sequential'``.

        """

        self.__color_limits(category)

        if self.im is not None:

            self.im.set_cmap(self.cmap)

    def create_colorbar(self, orientation='vertical', norm='linear'):
        """
        Create colorbar.

        Parameters
        ----------
        orientation : TYPE, optional
            Orientation of colorbar. Either ``'vertical'`` or ``'horizontal'``.
            The default is ``'vertical'``.
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

        self.remove_colorbar()

        pad = 0.05 if orientation.lower() == 'vertical' else 0.2

        self.cb = self.fig.colorbar(self.im, ax=self.ax,
                                    orientation=orientation, pad=pad)

        self.cb.ax.minorticks_on()

    def remove_colorbar(self):
        """
        Remove colorbar.

        """

        if self.im.colorbar is not None:

            self.im.colorbar.remove()

    def set_colorbar_label(self, label):
        """
        Update colorbar label.

        Parameters
        ----------
        label : str
            Colorbar label.

        """

        if self.im.colorbar is not None:

            self.im.colorbar.set_label(label)

    def reset_color_limits(self):
        """
        Autoscale color limits.

        """

        self.im.autoscale()

    def update_data(self, data, vmin, vmax):
        """
        Replace data.

        Parameters
        ----------
        data : 2d array
            Image values.
        vmin : float, optional
            Minimum data value.
        vmax : float, optional
            Maximum data value.

        """

        if self.im is not None:

            self.im.set_array(data.T)
            self.im.set_clim(vmin=vmin, vmax=vmax)

    def get_data(self):
        """
        Data.

        Returns
        -------
        data : 2d array
            Image values.

        """

        if self.im is not None:

            return self.im.get_array().T

    def plot_data(self, data, min_x, min_y, max_x, max_y, matrix=np.eye(2)):
        """
        Plot data.

        Parameters
        ----------
        data : 2d array
            Image values.
        min_x, min_y : float
            Minimumx extents.
        max_x, max_y : float
            Maxium extents.
        matrix : 2d array, 2x2, optional
            Transformation matrix. The default is the identity matrix.

        """

        size_x, size_y = data.shape[1], data.shape[0]

        extents = self.__extents(min_x, min_y, max_x, max_y, size_x, size_y)

        self.im = self.ax.imshow(data.T, interpolation='nearest',
                                 origin='lower', extent=extents)

        self.transform_axes(matrix)

        self.ax.minorticks_on()

    def transform_axes(self, matrix):
        """
        Skew axes according to a transformation matrix.

        Parameters
        ----------
        matrix : 2d array, 2x2
            Transformation matrix.

        """

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
        """
        Draw line with specified extents. If not specified, extents are over
        the entire axis limits.

        Parameters
        ----------
        xlim : list, optional
            Extents along x-axis. The default is ``None``.
        ylim : list, optional
            Extents along y-axis. The default is ``None``.

        Returns
        -------
        None.

        """

        if xlim is None: xlim = self.ax.get_xlim()
        if ylim is None: ylim = self.ax.get_ylim()

        self.ax.plot(xlim, ylim, color='w')

    def add_text(self, x, y, s, color='w'):
        """
        Add text at specified location.

        Parameters
        ----------
        x, y : float
            Text location.
        s : str
            Text string.
        color : str, optional
            Text color. The default is ``'w'```.

        """

        self.ax.text(x, y, s, color=color, ha='center', va='center')

    def add_grid_lines(self, alpha=0.0):

        self.ax.grid(which='both', alpha=alpha)

    def set_integer_axes(self):

        self.ax.xaxis.get_major_locator().set_params(integer=True)
        self.ax.yaxis.get_major_locator().set_params(integer=True)

class Scatter(Plot):
    """
    Intensity heat map plot.

    Parameters
    ----------
    canvas : canvas
        Canvas for embedding in GUI.

    Attributes
    ----------
    sc : scatter
        Scatter plot.
    norm : normalization
        Color normalization.
    cmap : colormap
        Colormap.

    Methods
    -------
    set_normalization()
        Update data normalization.
    update_normalization()
        Replace normalization.
    update_colormap()
        Replace colormap.
    create_colorbar()
        Create colorbar.
    remove_colorbar()
        Remove colorbar.
    set_colorbar_label()
        Update colorbar label.
    reset_color_limits()
        Autoscale color limits.
    update_data()
        Replace data.
    get_data()
        Data.
    plot_data()
        Plot data.

    """

    def __init__(self, canvas):

        super(Scatter, self).__init__(canvas)

        self.sc = None
        self.norm = None
        self.cmap = None

        self.__color_limits()

    def __color_limits(self, category='sequential'):

        if (category == 'sequential'):
            self.cmap = plt.cm.viridis
        elif (category == 'diverging'):
            self.cmap = plt.cm.bwr
        else:
            self.cmap = plt.cm.binary

    def set_normalization(self, vmin, vmax, norm='linear'):
        """
        Update data normalization.

        Parameters
        ----------
        vmin : float
            Minimum data value.
        vmax : float
            Maximum data value.
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

        if np.isclose(vmin, vmax): vmin, vmax = 1e-3, 1e+3

        if (norm.lower() == 'linear'):
            self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
        elif (norm.lower() == 'logarithmic'):
            self.norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            self.norm = colors.SymLogNorm(linthresh=0.1, linscale=0.9, base=10,
                                          vmin=vmin, vmax=vmax)

        self.sc.set_norm(self.norm)

        if self.sc.colorbar is not None:

            orientation = self.sc.colorbar.orientation

            self.remove_colorbar()
            self.create_colorbar(orientation, norm)

            if (norm.lower() == 'symlog'):
                sym_log = ticker.SymmetricalLogLocator(linthresh=0.1, base=10)
                self.cb.locator = sym_log
                subs = np.linspace(0.1,0.9,9)
                sym_log = ticker.SymmetricalLogLocator(linthresh=0.1, base=10,
                                                       subs=subs)
                self.cb.minorlocator = sym_log
                self.cb.formatter = ticker.ScalarFormatter()
                self.cb.update_ticks()

    def update_normalization(self, norm='linear'):
        """
        Replace normalization.

        Parameters
        ----------
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

        if self.sc is not None:

            vmin, vmax = self.sc.get_clim()

            self.set_normalization(vmin, vmax, norm)

    def update_colormap(self, category='sequential'):
        """
        Replace colormap.

        Parameters
        ----------
        category : str, optional
            Update colormap. Options are ``'sequential'``, ``'diverging'``, or
            ``'binary'``. The default is ``'sequential'``.

        """

        self.__color_limits(category)

        if self.sc is not None:

            self.sc.set_cmap(self.cmap)

    def create_colorbar(self, orientation='vertical', norm='linear'):
        """
        Create colorbar.

        Parameters
        ----------
        orientation : TYPE, optional
            Orientation of colorbar. Either ``'vertical'`` or ``'horizontal'``.
            The default is ``'vertical'``.
        norm : str, optional
            Data normalization. Options are ``'linear'``, ``'logarithmic'``, or
            ``'symlog'``. The default is ``'linear'``.

        """

        if self.sc is not None:

            self.remove_colorbar()

            pad = 0.05 if orientation.lower() == 'vertical' else 0.2

            self.cb = self.fig.colorbar(self.sc, ax=self.ax,
                                        orientation=orientation, pad=pad)

            self.cb.ax.minorticks_on()

    def remove_colorbar(self):
        """
        Remove colorbar.

        """

        if self.sc.colorbar is not None:

            self.sc.colorbar.remove()

    def set_colorbar_label(self, label):
        """
        Update colorbar label.

        Parameters
        ----------
        label : str
            Colorbar label.

        """

        if self.sc.colorbar is not None:

            self.sc.colorbar.set_label(label)

    def reset_colorbar_limits(self):
        """
        Autoscale color limits.

        """

        if self.sc is not None:

            self.sc.autoscale()

    def update_data(self, c, vmin, vmax):
        """
        Replace data.

        Parameters
        ----------
        c : 1d array
            Scatter point values.
        vmin : float, optional
            Minimum data value.
        vmax : float, optional
            Maximum data value.

        """

        if self.sc is not None:

            self.sc.set_array(c)
            self.sc.set_clim(vmin=vmin, vmax=vmax)

    def get_data(self):
        """
        Data.

        Returns
        -------
        c : 1d array
            Scatter point values.

        """

        if self.sc is not None:

            return self.sc.get_array()

    def plot_data(self, x, y, c):
        """
        Plot data.

        Parameters
        ----------
        x, y : 1d array
            Scatter point coordinates.
        c : 1d array
            Scatter point values.

        """

        self.sc = self.ax.scatter(x, y, c=c, cmap=self.cmap)
