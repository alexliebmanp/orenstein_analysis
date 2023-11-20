'''
helper.py

    Assortment of helper functions for use throughout the package.
'''
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def colormap_generator(min, max, cmap):
    '''
        returns a function for calculating a color based on a colormap and the bounds of the representative data. Also returns a scalar mappable

        args:
            - min: minimum value in colormap
            - max: maximum value in colormap
            - cmap: colormap object
        return:
            - colormap_func: function for evaluating color according to colormap
    '''

    colormap_func = lambda val: cmap(val*(1/(max - min)) - (min/(max - min)))

    norm = colors.Normalize(vmin=min,vmax=max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    return colormap_func, sm
