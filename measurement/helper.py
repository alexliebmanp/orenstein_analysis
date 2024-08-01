'''
helper.py

    Assortment of helper functions for use throughout the package.
'''
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import colorsys


def colormap_generator(min, max, cmap, cmap_frac=0):
    '''
        returns a function for calculating a color based on a colormap and the bounds of the representative data. Also returns a scalar mappable. Based on a linear mapping

        args:
            - min: minimum value in colormap
            - max: maximum value in colormap
            - cmap: colormap object
        return:
            - colormap_func: function for evaluating color according to colormap
    '''

    colormap_func = lambda val: cmap(val*((1-2*cmap_frac)/(max - min)) - ((1-2*cmap_frac)*min/(max - min)) + cmap_frac)

    norm = colors.Normalize(vmin=min,vmax=max)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    return colormap_func, sm

def adjust_color_hls(color, coordinate='l', amount=0.5):
    '''
    adjust color (input in any format) by hue 'h', saturation 's', or lightness 'l' by a specified amount
    '''
    c = colorsys.rgb_to_hls(*colors.to_rgb(color))
    if coordinate=='h':
        return colorsys.hls_to_rgb(max(0, min(1, amount * c[0])), c[1], c[2])
    elif coordinate=='l':
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    elif coordinate=='s':
        return colorsys.hls_to_rgb(c[0], c[1], max(0, min(1, amount * c[2])))
    else:
        return color


# Sum of the min & max of (a, b, c)
def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c
def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))