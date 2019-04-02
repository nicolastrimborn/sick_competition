import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import time as ti
#import pandas as pd
#import seaborn as sns
from open3d import *

import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll


def testplot(X0, Y0, Z0):
    xyz = np.zeros((np.size(X0),3))
    xyz[:,0] = X0
    xyz[:,1] = Y0
    xyz[:,2] = Z0

    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    write_point_cloud("../sync.ply", pcd)
    pcd_load = read_point_cloud("./sync.ply")
    downpcd = voxel_down_sample(pcd_load, voxel_size = 0.05)
    # draw_geometries([downpcd])
    estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
                            radius = 0.1, max_nn = 30))

    slope = vol.crop_point_cloud(downpcd)
    draw_geometries([slope])




    draw_geometries([downpcd])
    # draw_geometries([pcd_load])

def test_wirefreame(X0, Y0, Z0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X0, Y0, Z0, rstride=1, cstride=1)
    plt.show()


def test_points3d(X0, Y0, Z0):

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X0, Y0, Z0, cmap=plt.cm.ocean, linewidth=0.02, antialiased=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    # to Add a color bar which maps values to colors.
    # surf=ax.plot_trisurf(X0, Y0, Z0, cmap=plt.cm.ocean, linewidth=0.2)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_aspect('auto')
    # plt.show()

    # Rotate it
    # Other palette
    # ax.plot_trisurf(X0, Y0, Z0, cmap=plt.cm.ocean, linewidth=0.01)
    # plt.show()


def normalization(x, y, z):
    #Normalization in case of need it any calculation purpose
    xnorm = (x - np.min(x)) / (np.max(x) - np.min(x))
    ynorm = (y - np.min(y)) / (np.max(y) - np.min(y))
    znorm = (z - np.min(z)) / (np.max(z) - np.min(z))
    return xnorm, ynorm, znorm

def scatter_plot(X0, Y0, Z0, X1, Y1, Z1):
    X0 = np.ravel(X0)
    Y0 = np.ravel(Y0)
    Z0 = np.ravel(Z0)
    X1 = np.ravel(X1)
    Y1 = np.ravel(Y1)
    Z1 = np.ravel(Z1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, Y1, Z1, c='b', marker='^')
    ax.scatter(X0, Y0, Z0, c='r', marker='o')
    ax.set_title("Selected slope", fontweight='bold', fontsize=16, fontname='Arial', y=-0.05)
    ax.set_aspect('equal')
    plt.tight_layout()

    date = ti.strftime("%d_%m_%Y_%H_%M_%S")
    #plt.savefig('Slope_'+date+'.png', bbox_inches='tight')
    #plt.savefig('Slope.png', bbox_inches='tight')
    plt.show()


def surface_plot(X0, Y0, Z0, X1, Y1, Z1):
    X0 = np.ravel(X0)
    Y0 = np.ravel(Y0)
    Z0 = np.ravel(Z0)
    X1 = np.ravel(X1)
    Y1 = np.ravel(Y1)
    Z1 = np.ravel(Z1)
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 2, 1, projection='3d', adjustable='box')
    ax.view_init(30, 180)
    #bx = fig.gca(projection='3d')
    ax.plot_trisurf(X0, Y0, Z0, linewidth=0.5, antialiased=True)
    ax.set_title("Original slope", fontweight='bold', fontsize=16, fontname='Arial', y=-0.05)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(30, 180)
    ax.plot_trisurf(X1, Y1, Z1, linewidth=0.5, antialiased=True)
    ax.set_title("Selected slope", fontweight='bold', fontsize=16, fontname='Arial', y=-0.05)
    plt.tight_layout()
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
    smo=np.average(smoothIndex)
    print("Smoothness index: ",smo)
    return smo

def slopecalc(x, y, sensorpos):
    #Sensor position correction, sensor pos desceibes an offset coordinate
    y=y-sensorpos[1]
    x=np.ravel(x)
    y=np.ravel(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print("Slope: ", slope)
    return slope


if __name__ == '__main__':
    # Please provide surface data on 3 numpy arrays x, y and z
    x = np.array([[1, 1, 1, 2, 3],[4, 4, 4, 4, 1]])
    y = np.array([[1, 2, 3, 3, 3],[3, 4, 5, 6, 1]])
    z = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
    Y = np.array([[1, 1, 1, 2, 3, 4, 4, 4, 4, 1],[1, 2, 3, 3, 3, 3, 4, 5, 6, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # Defige sensor position to set reference frame on pole base (ground level)
    #[X,Y,Z]
    sensorpos = np.array([0, 3, 0])
    #x=np.reshape(x,())
    smo = smoothness(Y)
    surface_plot(x,y,z,x,y,z)
    # Calculate slope and smoothnes index from data provided
    slope = slopecalc(x, y, sensorpos)

    #NOTES:
    #If the slope have steps or is uneven, please select each one and repeat the calculation process (lines 67, 70)
