'''
data_methods.py

    Methods for processing data.
'''
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def process_corotate(measurement):
    '''
    fits a corotation scan to a sin(2theta) + sin(4theta) function.

    args:
        - measurement(Dataset):
    returns:
        - fits?
    '''
