'''
Define coordinate functions for use with define_coordinates()
'''

def corotation_coordinates(measurement):
    '''
    Returns a corotation angle
    '''

    return 'Polarization Angle (deg)', 2*measurement['Angle 1 (deg)'].data
