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


bag = rosbag.Bag('2019-03-12-19-53-04.bag','r')
Iter=0

for topic, msg, t in bag.read_messages(topics='/cloud'):
	Iter +=1
	print(Iter)
        x = np.array(list(pc2.read_points(msg, skip_nans=True, field_names = ("x"))))
        y = np.array(list(pc2.read_points(msg, skip_nans=True, field_names = ("y"))))
	z = np.array(list(pc2.read_points(msg, skip_nans=True, field_names = ("z"))))
	#cld = ros_numpy.numpify(msg, squeeze=False)
	print(x.shape)
	print(y.shape)
	print(z.shape)
	if Iter == 1:
		break
bag.close()

ncols=920
x = np.reshape(x, (-1, ncols))
y = np.reshape(y, (-1, ncols))
z = np.reshape(z, (-1, ncols))

x= x[:,505:605]
y= y[:,505:605]
z= z[:,505:605]

x=x.T
y=y.T
z=z.T
print(x.shape)
print(y.shape)
print(z.shape)

xn=tile(array(x[:,0]), (24,1))
xn=xn.T
print(xn.shape)

fig, ax = plt.subplots()
min_val, max_val = 0, 15

intersection_matrix = x - xn

#cm = mpl.colors.Colormap(Greys,10)
#ax.matshow(intersection_matrix, cmap=plt.cm.Greys_r)
ax.matshow(intersection_matrix, cmap=plt.cm.get_cmap('Greys_r',10))
ax.set_aspect(aspect='auto', adjustable='box')
raw_input("Press enter to continue")
