from flask import Flask, render_template, request
import json
from Analysis import slopecalc
from Analysis import smoothness
from Analysis import surface_plot
from Analysis import test_points3d
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from pylab import *
import ros_numpy
global umin
global umax
global vmin
global vmax
umin = 400
umax = 600
vmin = 0
vmax = 24
import thread

"""start of flask application"""
app = Flask(__name__)


@app.route('/<string:page_name>/')
def static_page(page_name):
    return render_template('%s.html' % page_name)


@app.route('/')
def home_page():
    return render_template('test.html')


@app.route('/slope_val', methods=['GET'])
def send_slope_val():
    global slope_val
    print("sending slope val: {}".format(slope_val))
    slope_val = {"Slope": slope_val}
    slopey = json.dumps(slope_val)
    return slopey


@app.route('/smoothness_val', methods=['GET'])
def send_smoothness_val():
    global smoothness_val
    print("sending smoothness: {}".format(smoothness_val))
    smoothness_val = {"Smoothness":smoothness_val}
    bumpy = json.dumps(smoothness_val)
    return bumpy


@app.route('/range_layers', methods=['GET'])
def get_range_layer():
    global layers_to
    global layers_from
    global vmin
    global vmax
    layers_from = request.args.get('layers-from')
    layers_to = request.args.get('layers-to')
    vmin = int(layers_from)
    vmax = int(layers_to)
    # layers_selected = json.dumps(layers)
    print("my layer vals are: {} {}".format(layers_from,layers_to))
    return "ok"


@app.route('/range_points', methods=['GET'])
def get_range_point():
    global points_to
    global points_from
    global umin
    global umax
    points_from = request.args.get('points-from')
    points_to = request.args.get('points-to')
    umin = int(points_from)
    umax = int(points_to)
    print("my point vals are: {} {}".format(points_from,points_to))
    return "ok"


"""end of flask application"""

"""start of ROS Subscriptions"""


def subscribePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    thread.start_new_thread(flask_Thread, ())
    msg= rospy.Subscriber('/cloud', PointCloud2, processPointCloud2)
    rospy.spin()


def processPointCloud2(msg):
    global slope_val
    global smoothness_val
    global umin
    global umax
    global vmin
    global vmax

    # Convert PointCloud2 to np.Array
    cld = ros_numpy.numpify(msg, squeeze=False)

    xvals = cld[vmin:vmax, umin:umax]['x'].ravel()
    yvals = cld[vmin:vmax, umin:umax]['y'].ravel()
    zvals = cld[vmin:vmax, umin:umax]['z'].ravel()
    intsvals = cld[vmin:vmax, umin:umax]['intensity'].ravel()
    sensorpos = np.array([0, 0, 0])

    cld1 = cld[cld['intensity'] > 64]
    x0 = cld1['x'].ravel()
    y0 = cld1['y'].ravel()
    z0 = cld1['z'].ravel()

    test_points3d(x0, z0, -y0)

    start = 525
    #skiPathComparison(x, y, z, start, False)
    #idealSurfaceComparision(x, y, z, start, False)
    # Initialise data structure to publish subset data
    data = np.zeros(np.shape(xvals), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)
    ])

    data['x'] = xvals
    data['y'] = yvals
    data['z'] = zvals
    data['intensity'] = intsvals
    slope_val = slopecalc(cld[vmin:vmax, umin:umax]['x'], cld[vmin:vmax, umin:umax]['y'], sensorpos)
    smoothness_val = smoothness(cld['y'])
    smoothness_val = str(smoothness_val)
    surface_plot(x0, y0, z0, xvals, yvals, zvals)


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

    if not update:
        x1 = np.zeros((100,24))

    if update:
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

    if not update:
        x1 = np.zeros((100,24))

    if update:
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


def flask_Thread():
    app.run("0.0.0.0", port=4444)


if __name__ == '__main__':
    subscribePointCloud2FromSick()

