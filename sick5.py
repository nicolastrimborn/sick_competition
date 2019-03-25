#!/usr/bin/env python
from roslib import message
import rospy
import rosbag
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from Analysis import slopecalc
import numpy as np
from scipy.spatial.transform import Rotation as R

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

    ''' Bumpy area 2019-03-12-19-51-09.bag
    # Range of points for each layer
    umin= 572
    umax= 588
    # Range of layers
    vmin=6
    vmax=21
    '''

    ''' Bumpy area 2019-03-12-19-51-09.bag '''
    # Range of points for each layer
    umin= 0
    umax= 920
    # Range of layers
    vmin=0
    vmax=24


    cld= ros_numpy.numpify(msg, squeeze=False)

    # n = 24*920
    # datacopy = np.zeros((n,3))
    # datacopy[:,0] = cld['x'].ravel()
    # datacopy[:,1] = cld['y'].ravel() - 3
    # datacopy[:,2] = cld['z'].ravel()
    # r = R.from_euler('x', -90, degrees=True)
    # rotatedData = r.apply(datacopy)

    # print(cld['x'])
    # cld = r.apply([cld['x'].ravel(), cld['y'].ravel(), cld['z'].ravel()])

    xvals= cld[vmin:vmax, umin:umax]['x'].ravel()
    yvals= cld[vmin:vmax, umin:umax]['y'].ravel()
    zvals= cld[vmin:vmax, umin:umax]['z'].ravel()
    intsvals =  cld[vmin:vmax, umin:umax]['intensity'].ravel()
    sensorpos = np.array([0, 0, 0])

    data = np.zeros(np.shape(xvals), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)
    ])

    # data['x'] = xvals
    # data['y'] = yvals
    # data['z'] = zvals

    data['x'] =  rotatedData[:,0]
    data['y'] = rotatedData[:,1]
    data['z'] = rotatedData[:,2]
    data['intensity'] = intsvals

    publishAsPointcloud2(data,'/subcloud')

def publishAsPointcloud2(data, topic):

    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header.frame_id = 'laser'
    pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
    pub.publish(msg)

def playBag():
    path = '/home/nick/sick_competition/2019-03-12-19-51-09.bag'
    bag = rosbag.Bag(path)
    for topic, msg, t in bag.read_messages():
        print topic
        publishAsPointcloud2(ros_numpy.numpify(msg, squeeze=False),topic)
    bag.close()
    # subscribePointCloud2FromSick()

if __name__ == '__main__':
    try:
        # playBag()
        subscribePointCloud2FromSick()
    except rospy.ROSInterruptException:
        pass
