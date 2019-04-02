#!/usr/bin/env python
from roslib import message
import rospy
import rosbag
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from Analysis import slopecalc
from Analysis import smoothness
from Analysis import surface_plot
from Analysis import scatter_plot
from Analysis import test_points3d
import numpy as np


import rosbag
from roslib import message
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import matplotlib as mpl
import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl         # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
from pylab import *              # Matplotlib's pylab interface
ion()                            # Turned on Matplotlib's interactive mode

import ros_numpy
from rospy.numpy_msg import numpy_msg
from matplotlib.colors import LinearSegmentedColormap


# from scipy.spatial.transform import Rotation as R

def getOnePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    msg= rospy.wait_for_message('/cloud', PointCloud2)
    processPointCloud2(msg)

def subscribePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    msg= rospy.Subscriber('/cloud', PointCloud2, processPointCloud2)
    rospy.spin()

def processPointCloud2(msg):

    ''' Bumpy area 2019-03-12-19-51-09.bag
    # Range of points for each layer
    umin= 516
    umax= 560
    # Range of layers
    vmin=0
    vmax=18
    '''

    ''' Bumpy area 2019-03-12-19-51-09.bag'''
    # Range of points for each layer
    umin= 572
    umax= 588
    # Range of layers
    vmin=6
    vmax=21

    ''' Bumpy area 2019-03-12-19-51-09.bag 
    # Range of points for each layer
    umin= 0
    umax= 920
    # Range of layers
    vmin=0
    vmax=24
    '''

    # Convert PointCloud2 to np.Array
    cld = ros_numpy.numpify(msg, squeeze=False)

    x= cld['x'].ravel()
    y= cld['y'].ravel()
    z= cld['z'].ravel()

    cld1 = cld[cld['intensity'] > 64]
    x0= cld1['x'].ravel()
    y0= cld1['y'].ravel()
    z0= cld1['z'].ravel()

    test_points3d(x0, z0, -y0)

    start= 525
    skiPathComparison(x, y, z, start, False)
    idealSurfaceComparision(x, y, z, start, False)
    # Initialise data structure to publish subset data
    # data = np.zeros(np.shape(xvals), dtype=[
    #     ('x', np.float32),
    #     ('y', np.float32),
    #     ('z', np.float32),
    #     ('intensity', np.float32)
    # ])
    #
    # data['x'] = xvals
    # data['y'] = yvals
    # data['z'] = zvals
    # data['intensity'] = intsvals

    # slopecalc(xvals, yvals, sensorpos)
    # smoothness(cld['y'])
    # surface_plot(x0, y0, z0, xvals, yvals, zvals)
    # publishAsPointcloud2(data,'/subcloud')
    # scatter_plot(x0, z0, -y0, xvals, zvals, -yvals)


def publishAsPointcloud2(data, topic):
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header.frame_id = 'laser'
    pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
    pub.publish(msg)

def skiPathComparison(x, y, z, start, update):
    ncols=920
    x = np.reshape(x, (-1, ncols))
    y = np.reshape(y, (-1, ncols))
    z = np.reshape(z, (-1, ncols))

    x= x[:,start : start + 100]
    y= y[:,start : start + 100]
    z= z[:,start : start + 100]

    x=x.T
    y=y.T
    z=z.T

    xn=tile(array(x[:,0]), (24,1))
    xn=xn.T

    if update == False:
        x1 = np.zeros((100,24))

    if update == True:
        x1 = xn

    fig, ax = plt.subplots()
    intersection_matrix = x - x1
    cs=ax.matshow(intersection_matrix, cmap=plt.cm.get_cmap('Greys_r',10))
    ax.set_aspect(aspect='auto', adjustable='box')
    ax.axis('off')
    cbar = fig.colorbar(cs)
    cbar.set_label('surface disturbance (m)')
    plt.savefig("track.png")
    plt.close(fig)

def idealSurfaceComparision(x, y, z, start, update):
    ncols=920
    x = np.reshape(x, (-1, ncols))
    y = np.reshape(y, (-1, ncols))
    z = np.reshape(z, (-1, ncols))

    x= x[:,start : start + 100]
    y= y[:,start : start + 100]
    z= z[:,start : start + 100]

    x=x.T
    y=y.T
    z=z.T

    x_ = np.zeros(shape=x.shape)
    for i in range(24):
        x_[:,i]=np.linspace(x[0,i], x[-1,i], num=100)

    if update == False:
        x1 = np.zeros((100,24))

    if update == True:
        x1 = x_

    fig1, ax1 = plt.subplots()
    intersection_matrix = x - x1
    cs1 = ax1.matshow(intersection_matrix, cmap=plt.cm.get_cmap('Greys_r',10))
    ax1.set_aspect(aspect='auto', adjustable='box')
    cbar1 = fig1.colorbar(cs1)
    ax1.axis('off')
    cbar1.set_label('surface depth (m)')
    plt.savefig("surface.png")
    plt.close(fig1)

def playBag():
    path = '/hopme/nick/sick_competition/2019-03-12-19-51-09.bag'
    bag = rosbag.Bag(path)
    for topic, msg, t in bag.read_messages():
        print topic
        publishAsPointcloud2(ros_numpy.numpify(msg, squeeze=False),topic)
    bag.close()
    # subscribePointCloud2FromSick()

def rotate(matrix):
    # datacopy = np.zeros((len(cld['x'].ravel()),3))
    # datacopy[:,0] = cld['x'].ravel()
    # datacopy[:,1] = cld['y'].ravel() - 3
    # datacopy[:,2] = cld['z'].ravel()
    r = R.from_euler('x', -90, degrees=True)
    print(matrix)
    print((matrix[0]))
    # r.apply(matrix[0,:], matrix[1,:], matrix[2,:])
    # return matrix[0], matrix[1], matrix[2]


if __name__ == '__main__':
    try:
        # playBag()
        subscribePointCloud2FromSick()
    except rospy.ROSInterruptException:
        pass
