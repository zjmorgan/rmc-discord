import os

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.pyplot as plt

no_qt = os.environ.get('QT_QPA_PLATFORM') == 'offscreen'

FigureCanvas = FigureCanvasAgg if no_qt else FigureCanvasQTAgg

class Canvas(FigureCanvas):

    def __init__(self, parent=None):

        self.figure = plt.figure()
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
