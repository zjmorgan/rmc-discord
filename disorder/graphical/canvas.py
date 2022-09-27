from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import matplotlib.pyplot as plt

class Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = plt.figure()
        FigureCanvasQTAgg.__init__(self, self.figure)
        self.setParent(parent)
