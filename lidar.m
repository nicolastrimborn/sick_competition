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

start= 505
x= x[:,start : start + 100]
y= y[:,start : start + 100]
z= z[:,start : start + 100]

x=x.T
y=y.T
z=z.T


###########
xn=tile(array(x[:,0]), (24,1))
xn=xn.T
fig, ax = plt.subplots()
intersection_matrix = x - xn
cs=ax.matshow(intersection_matrix, cmap=plt.cm.get_cmap('Greys_r',10))
ax.set_aspect(aspect='auto', adjustable='box')
ax.axis('off')
cbar = fig.colorbar(cs)
cbar.set_label('surface depth (m)')
plt.savefig("track.png")
plt.close(fig)

#raw_input("Press enter to continue")


###########
x_ = np.zeros(shape=x.shape)
for i in range(24):
	x_[:,i]=np.linspace(x[0,i], x[-1,i], num=100)

print(x_.shape)
fig1, ax1 = plt.subplots()
intersection_matrix = x - x_
cs1 = ax1.matshow(intersection_matrix, cmap=plt.cm.get_cmap('Greys_r',10))
ax1.set_aspect(aspect='auto', adjustable='box')
cbar1 = fig1.colorbar(cs1)
ax1.axis('off')
cbar1.set_label('surface depth (m)')
plt.savefig("surface.png")
plt.close(fig1)

#raw_input("Press enter to continue")
