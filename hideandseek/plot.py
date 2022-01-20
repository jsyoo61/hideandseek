import itertools as it

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tools as T
import tools.numpy

def plot_values(valuetracker_list, ax=None):
    n_line = len(valuetracker_list)
    if ax is None:
        fig, ax = plt.subplots()
    color_list = it.cycle(mcolors.TABLEAU_COLORS)
    for valuetracker, color in zip(valuetracker_list, color_list):
        y_smooth = T.numpy.moving_mean(valuetracker.y, 9)
        ax.plot(valuetracker.x, valuetracker.y, color=color, alpha=0.4)
        ax.plot(valuetracker.x, y_smooth, color=color)
    return ax
