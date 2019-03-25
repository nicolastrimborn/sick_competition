import numpy as np
from scipy import stats

def normalization(x, y, z):
    #Normalization in case of need it any calculation purpose
    xnorm = (x - np.min(x)) / (np.max(x) - np.min(x))
    ynorm = (y - np.min(y)) / (np.max(y) - np.min(y))
    znorm = (z - np.min(z)) / (np.max(z) - np.min(z))
    return xnorm, ynorm, znorm

def slopecalc(x, y, z, sensorpos):
    #Sensor position correction
    z=z-sensorpos[2]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
    smoothness = np.average(np.gradient(x))
    #Gradient gives 1 if change is smooth. Average gradient creates an index where 1 is the most smooth, >1 bumpy and <1 holey
    print("Slope: ", slope,", Smothness: ", smoothness)
    return slope, smoothness


if __name__ == '__main__':
    # Please provide surface data on 3 numpy arrays x, y and z
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    z = np.array([1, 2, 3, 4, 5])
    # Defige sensor position to set reference frame on pole base (ground level)
    sensorpos = np.array([0, 0, 3])
    # Calculate slope and smoothnes index from data provided
    slope, smoothness = slopecalc(x, y, z, sensorpos)

    #NOTES:
    #If the slope have steps or is uneven, please select each one and repeat the calculation process (line 29)