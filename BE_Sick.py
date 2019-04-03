#******************************
#           RINNETUTKA
#   TAMPERE UNIVERSITY - FASTLAB
#   SICK INNOVATION COMPETITION 2019
#            BACKEND
#******************************

#******************************
#Libraries
#******************************

from flask import Flask, render_template, request
import json
from Analysis import slopecalc
from Analysis import smoothness
from Analysis import surface_plot
from Analysis import skiPathComp
from Analysis import idealSurfaceComp
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from pylab import *
import ros_numpy
import thread
#from Analysis import test_points3d

# Declare global variables
global umin
global umax
global vmin
global vmax
# Initialize variables
umin = 400
umax = 600
vmin = 0
vmax = 24

#******************************
#Back End for Web Interface
#******************************
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


#******************************
#ROS and Data Analytics
#******************************

def subscribePointCloud2FromSick():
    rospy.init_node('sick_mrs_6xxx', anonymous=True)
    thread.start_new_thread(flask_Thread, ())
    msg= rospy.Subscriber('/cloud', PointCloud2, processPointCloud2)
    rospy.spin()


def processPointCloud2(msg):
    # Define common variables with frontend
    global slope_val
    global smoothness_val
    global umin
    global umax
    global vmin
    global vmax

    # Convert PointCloud2 to np.Array
    cld = ros_numpy.numpify(msg, squeeze=False)

    # Create a new cloud with High Intensity (HI) points - Ignore empty areas
    cldHI = cld[cld['intensity'] > 64]
    xHI = cldHI['x'].ravel()
    yHI = cldHI['y'].ravel()
    zHI = cldHI['z'].ravel()

    # Extract section selected on frontend by sliders
    xvals = cld[vmin:vmax, umin:umax]['x']
    yvals = cld[vmin:vmax, umin:umax]['y']
    zvals = cld[vmin:vmax, umin:umax]['z']

    # Ravel section for ploting
    xrav = xvals.ravel()
    yrav = yvals.ravel()
    zrav = zvals.ravel()
    intsvals = cld[vmin:vmax, umin:umax]['intensity'].ravel()

    # Indicate sensor position for height correction
    sensorpos = np.array([0, 0, 0])

    # Calculate slope of extracted section
    slope_val = slopecalc(xvals, yvals, sensorpos)

    # Calculate smoothness of extracted section
    smoothness_val = smoothness(yvals)

    # String??
    smoothness_val = str(smoothness_val)

    # Generate frontend images of whole slope  and extracted section
    surface_plot(xHI, yHI, zHI, xrav, yrav, zrav)

    #start = 525
    skiPathComp(x, y, z, start, False)
    idealSurfaceComp(x, y, z, start, False)

    #test plot
    #test_points3d(x0, z0, -y0)

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

#Not used
def publishAsPointcloud2(data, topic):

    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header.frame_id = 'laser'
    pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
    pub.publish(msg)

#******************************
#Initialization
#Main thread: ROS
#Second thread: Flask server
#******************************


def flask_Thread():

    app.run("0.0.0.0", port=4444)


if __name__ == '__main__':

    subscribePointCloud2FromSick()