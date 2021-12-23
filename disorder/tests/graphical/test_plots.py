#!/usr/bin/env python3

import unittest

import numpy as np
np.random.seed(13)

from disorder.graphical.canvas import Canvas
from disorder.graphical.plots import Plot, Line, Scatter, HeatMap

import os
directory = os.path.dirname(os.path.abspath(__file__))

class test_plots(unittest.TestCase):

    def setUp(self):

        self.canvas = Canvas()

    def tearDown(self):

        self.canvas.close()

    def test_plot(self):

        base_plot = Plot(self.canvas)

        filename = os.path.join(directory, 'plot.png')
        base_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))

        os.remove(filename)

        base_plot.clear_canvas()
        base_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_line(self):

        line_plot = Line(self.canvas)

        x = np.linspace(0,1,16)

        ys = np.sin(2*np.pi*x)

        line_plot.plot_data(x, ys, marker='o', label=r'$\sin(2 \pi x)$')

        yc = np.cos(2*np.pi*x)

        line_plot.plot_data(x, yc, marker='s', label=r'$\cos(2 \pi x)$')

        line_plot.show_legend()

        np.testing.assert_array_equal(line_plot.p[0].lines[0].get_ydata(), ys)
        np.testing.assert_array_equal(line_plot.p[1].lines[0].get_ydata(), yc)

        x, ys = line_plot.get_data(0)
        x, yc = line_plot.get_data(1)

        line_plot.update_data(2*x, np.sin(2*np.pi*x), 0)
        line_plot.update_data(2*x, np.cos(2*np.pi*x), 1)

        line_plot.reset_view()
        line_plot.clear_lines()

        X = np.linspace(-1,1,256)
        Y = np.arctan(2*np.pi*X)

        line_plot.plot_data(X, Y, marker='-', label=r'$y=f(x)$')

        x = np.linspace(-1,1,16)
        y = np.arctan(2*np.pi*x)

        yerr = 0.1*np.abs(y)

        line_plot.plot_data(x, y, yerr=yerr, marker='o', label=r'$y_i=f(x_i)$')

        line_plot.reset_view()
        line_plot.show_legend()
        line_plot.set_labels(r'$y=f(x)$', r'$x$', r'$y$')

        filename = os.path.join(directory, 'line.png')
        line_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        line_plot.clear_canvas()

        x = np.linspace(-1,1,16)
        y = np.arctan(2*np.pi*x)*2/np.pi
        yerr = 0.1*np.abs(y)

        line_plot.plot_data(x, y, yerr=yerr, marker='o', label=r'y')
        line_plot.set_normalization('symlog')
        line_plot.set_limits(-1, 1)
        line_plot.draw_horizontal()

        z = 0.5*(y+1)
        zerr = 0.1*np.abs(z)

        line_plot.plot_data(x, z, yerr=zerr, marker='s', label=r'z', twin=True)
        line_plot.set_normalization('symlog', twin=True)
        line_plot.set_limits(0, 1, twin=True)

        line_plot.show_legend()
        line_plot.set_labels(r'$y=f(x)$', r'$x$', r'$y$', r'$z=(y+1)/2$')

        line_plot.set_aspect(0.75)

        line_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_heatmap(self):

        heat_map = HeatMap(self.canvas)

        xmin, xmax, xsize = 0, 1, 16
        ymin, ymax, ysize = 2, 4, 32

        x, y = np.meshgrid(np.linspace(xmin, xmax, xsize),
                           np.linspace(ymin, ymax, ysize), indexing='ij')

        z = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

        heat_map.plot_data(z, xmin, ymin, xmax, ymax)
        heat_map.create_colorbar(orientation='horizontal')
        heat_map.update_colormap('binary')

        np.testing.assert_array_equal(heat_map.get_data(), z)

        filename = os.path.join(directory, 'heatmap.png')
        heat_map.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        heat_map.set_normalization(-1, 1, norm='symlog')
        heat_map.update_colormap(category='diverging')
        heat_map.reformat_colorbar()

        heat_map.set_normalization(-1.1, 1.1, norm='symlog')
        heat_map.reformat_colorbar()

        heat_map.set_normalization(-0.4, 0.4, norm='symlog')
        heat_map.reformat_colorbar()

        heat_map.set_labels(r'$z(x,y)$', r'$x$', r'$y$')

        heat_map.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        z = 1000*np.exp(-2*x)*np.exp(-y)

        heat_map.update_data(z, 1, 1000)
        heat_map.update_normalization('logarithmic')
        heat_map.update_colormap(category='sequential')

        transform = np.array([[1.0,0.5],[-0.5,1.0]])
        heat_map.transform_axes(transform)

        heat_map.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_scatter(self):

        scatter_plot = Scatter(self.canvas)

        size = 64
        xmin, xmax = 0, 1
        ymin, ymax, = 2, 4

        x = np.random.uniform(xmin, xmax, size)
        y = np.random.uniform(ymin, ymax, size)

        c = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

        scatter_plot.plot_data(x, y, c)
        scatter_plot.create_colorbar()
        scatter_plot.update_colormap('binary')

        np.testing.assert_array_equal(scatter_plot.get_data(), c)

        filename = os.path.join(directory, 'scatter.png')
        scatter_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        scatter_plot.set_normalization(-1, 1, norm='symlog')
        scatter_plot.update_colormap(category='diverging')
        scatter_plot.reformat_colorbar()

        scatter_plot.set_normalization(-1.1, 1.1, norm='symlog')
        scatter_plot.reformat_colorbar()

        scatter_plot.set_normalization(-0.4, 0.4, norm='symlog')
        scatter_plot.reformat_colorbar()

        scatter_plot.set_labels(r'$f(x,y)$', r'$x$', r'$y$')

        scatter_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

        c = 1000*np.exp(-2*x)*np.exp(-y)

        scatter_plot.update_data(c, 1, 1000)
        scatter_plot.update_normalization('logarithmic')
        scatter_plot.update_colormap(category='sequential')

        scatter_plot.save_figure(filename)

        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()