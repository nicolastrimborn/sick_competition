#!/usr/bin/env python
from roslib import message
import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from analysis import slopecalc
import numpy as np
import time

#listener

def getOnePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    msg= rospy.wait_for_message('/cloud', PointCloud2)
    processPointCloud2(msg)

def subscribePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    msg= rospy.Subscriber('/cloud', PointCloud2, processPointCloud2)
    print(msg)
    rospy.spin()

def processPointCloud2(pointCloud):
    #rospy.init_node('sick_mrs_6xxx', anonymous=True)
    msg= pointCloud
    points=[]
    # Range of points for each layer
    umin= 519
    umax= 590
    # Range of layers
    vmin=19
    vmax=24
    #NP array of points [row,col,[x,y,z,i]]
    t = time.time()
    cld= ros_numpy.numpify(pointCloud, squeeze=False)
     
    t1 = time.time()
            
    print(t1-t)
    print(np.shape(cld['x'].ravel()))
    #print(cld)
    xvals= cld[vmin:vmax, umin:umax]['x'].ravel()
    yvals= cld[vmin:vmax, umin:umax]['y'].ravel()
    zvals= cld[vmin:vmax, umin:umax]['z'].ravel()
    sensorpos = np.array([0, 0, 0])
    print('xvals:'+str(xvals))
    print('yvals:'+str(yvals))
    print('zvals:'+str(zvals))
    slope, smoothness = slopecalc(xvals, yvals, zvals, sensorpos)
    print('slope:' + str(slope))
    print('smoothness: '+ str(smoothness))

def publishAsPointcloud2(points, topic, header, fields):
    for u in range(920):
        for v in range(24):
            x, y, z, i= cld[u, v, :]
            points.append([x, y, z, i])

    cloud= pc2.create_cloud(header,fields, points=points)
    pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(cloud)
        rate.sleep()

def callback_kinect(data) :
    rospy.loginfo("int_data " + str(data))
    

if __name__ == '__main__':
    try:
        subscribePointCloud2FromSick()
    except rospy.ROSInterruptException:
        pass
