import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time as ti

def normalization(x, y, z):
    #Normalization in case of need it any calculation purpose
    xnorm = (x - np.min(x)) / (np.max(x) - np.min(x))
    ynorm = (y - np.min(y)) / (np.max(y) - np.min(y))
    znorm = (z - np.min(z)) / (np.max(z) - np.min(z))
    return xnorm, ynorm, znorm

def surface_plot(X,Y,Z):
    #X = np.ravel(X)
    #Y = np.ravel(Y)
    #Z = np.ravel(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.5, antialiased=True)
    date = ti.strftime("%d_%m_%Y_%H_%M_%S")
    plt.savefig('Slope_'+date+'.png', bbox_inches='tight')
    plt.savefig('Slope.png', bbox_inches='tight')
    plt.show()

def smoothness(Y):
    # Gradient gives 1 if change is smooth.
    # Average gradient creates an index where 1 is the most smooth,
    # >1 bumpy, <1 holey, <0 slope inverted
    smoothIndex=[]
    for i in range(Y.shape[1]):
        smoothness = np.average(np.gradient(Y[:,i]))
        smoothIndex.append(smoothness)
    return np.average(smoothIndex)

def slopecalc(x, y, sensorpos):
    #Sensor position correction, sensor pos desceibes an offset coordinate
    y=y-sensorpos[2]
    x=np.ravel(x)
    y=np.ravel(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print("Slope: ", slope)
    return slope


if __name__ == '__main__':
    # Please provide surface data on 3 numpy arrays x, y and z
    x = np.array([1, 1, 1, 2, 3, 4, 4, 4, 4, 1])
    y = np.array([1, 2, 3, 3, 3, 3, 4, 5, 6, 1])
    z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Y = np.array([[1, 1, 1, 2, 3, 4, 4, 4, 4, 1],[1, 2, 3, 3, 3, 3, 4, 5, 6, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # Defige sensor position to set reference frame on pole base (ground level)
    sensorpos = np.array([0, 0, 3])
    #x=np.reshape(x,())
    #smo=smoothness(Y)
    #print("Smoothness: ", smo)
    #surface_plot(x,y,z)
    # Calculate slope and smoothnes index from data provided
    slope = slopecalc(x, y, sensorpos)

    #NOTES:
    #If the slope have steps or is uneven, please select each one and repeat the calculation process (line 29)
