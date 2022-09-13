import numpy as np

def statistics(self, data):

    return np.mean(data, axis=0), np.std(data, axis=0)
