import re

from disorder.material import tables

from disorder.graphical.canvas import Canvas
from disorder.graphical.plots import Scatter
from disorder.graphical import colors

Z = tables.Z
bc = tables.bc

no, sigma = [], []
for key in bc.keys():
    b, key = bc[key], re.sub(r'(^|\W)\d+', '', key)
    z = Z[key]
    sigma.append(4*np.pi*np.abs(b)**2)
    no.append(z)

canvas = Canvas()

scatter = Scatter(canvas)
scatter.plot_data(no, sigma, no)
scatter.set_labels(r'', r'$z$', r'$\sigma_c$')
scatter.set_axis_scales('linear', 'log')
scatter.sc.set_cmap(colors.elements)

canvas.close()
