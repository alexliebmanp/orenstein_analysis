'''
helper.py

    Assortment of helper functions for use throughout the package.
'''

def colormap_generator(min, max, cmap):
    '''
        returns a function for calculating a color based on a colormap and the bounds of the representative data.

        args:
            - min: minimum value in colormap
            - max: maximum value in colormap
            - cmap: colormap
        return:
            - colormap_func: function for evaluating color according to colormap
    '''

    colormap_func = lambda val: cmap(val*(1/(max - min)) - (min/(max - min)))
    return colormap_func
