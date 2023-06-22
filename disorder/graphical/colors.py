#!/usr/bin/env python3

import numpy as np
from matplotlib.colors import ListedColormap

from disorder.material import tables

def element_colormap():
    Z = tables.Z
    rgb = tables.rgb

    color_values = []
    for key in Z.keys():
        color = rgb.get(key)
        if color is None:
            color = rgb['XX']
        color_values.append(color)

    color_values = np.array(color_values)

    return ListedColormap(color_values)

elements = element_colormap()