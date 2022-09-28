import numpy as np
import matplotlib.pyplot as plt

from disorder.graphical.canvas import Canvas
from disorder.graphical.plots import Line

plt.switch_backend('agg')
plt.ioff()

x = np.linspace(0,1,16)
y = np.sin(2*np.pi*x)

X = np.linspace(0,1,128)
Y = np.cos(2*np.pi*X)

canvas = Canvas()

line = Line(canvas)
line.plot_data(x, y, marker='o', label=r'$\sin(2 \pi x)$')
line.plot_data(X, Y, marker='-', label=r'$\cos(2 \pi x)$')
line.set_labels(r'$y=f(x)$', r'$x$', r'$y$')
line.show_legend()
