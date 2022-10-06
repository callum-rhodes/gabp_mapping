import numpy as np
import rospy.rostime
import tf2_ros


class GasSensor:
    def __init__(self, type=1, sensor_topic=None, sensor_msg_type=None, tf_parent=None, tf_child=None):
        if type == 1:
            self.units = "ppb"
            self.multiplier = 1e6
        elif type == 2:
            self.units = "ppm"
            self.multiplier = 1e3
        elif type == 3:
            self.units = "kg/m^3"
            self.multiplier = 1
        else:
            self.units = "unknown"
            self.multiplier = 1

        self.variance = (7 ** 2) / self.multiplier
        self.misdetect = 0.1

        self.data = []
        self.pose = []
        self.timestamp = []

        if sensor_topic is not None:
	    self.sensor_topic = sensor_topic
	    self.sensor_msg_type = sensor_msg_type
            self.rate = rospy.Rate(10)
            self.sub = rospy.Subscriber(sensor_topic, sensor_msg_type, self.sensor_callback, queue_size=1)
	    self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
	    self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
            self.parent_frame = tf_parent
            self.child_frame = tf_child

    def get_conc(self, map, pos, time=0):
        pos_map_x = round(pos[0] / map.gridsize)
        pos_map_y = round(pos[1] / map.gridsize)
        conc_gt = map.data[pos_map_y, pos_map_x]
        conc = np.random.normal(conc_gt, self.variance)
        conc = max(conc, 0)
        misdetection = np.random.rand() <= self.misdetect
        if misdetection is True:
            conc = 0

        conc = conc * self.multiplier

        self.data.append(conc)
        self.pose.append(pos)
        self.timestamp.append(time)

    def insert_conc(self, conc, pos, time=0):
        self.data.append(conc)
        self.pose.append(pos)
        self.timestamp.append(time)

    def sensor_callback(self, data):
        try:
            # self.tf_listener.waitForTransform(self.parent_frame, self.child_frame, rospy.Time(0), rospy.Duration(1))
            # (pose, rot) = self.tf_listener.lookupTransform(self.parent_frame, self.child_frame, rospy.Time(0))
	    trans = self.tfBuffer.lookup_transform(self.parent_frame, self.child_frame, rospy.Time(0))
            self.timestamp.append(rospy.get_time())
            self.pose.append([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            self.data.append(data.data)
            # self.rate.sleep()

        except(tf2_ros.LookupException):
            rospy.logwarn('!!! Sensor TF lookup error, attempting to reinitialise sensor until TF name found !!!')
	    self.sub.unregister()
	    del self.tf_listener
	    del self.tfBuffer
	    self.sub = rospy.Subscriber(self.sensor_topic, self.sensor_msg_type, self.sensor_callback, queue_size=1)
	    self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
	    self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        except(tf2_ros.ConnectivityException):
            rospy.logwarn('!!! Sensor TF cannot find connection between requested TFs, attempting to reinitialise sensor until connection found !!!')
	    self.sub.unregister()
	    del self.tf_listener
	    del self.tfBuffer
	    self.sub = rospy.Subscriber(self.sensor_topic, self.sensor_msg_type, self.sensor_callback, queue_size=1)
	    self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
	    self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        except(tf2_ros.ExtrapolationException):
            rospy.logwarn('!!! Sensor TF extrapolation error, attempting to reinitialise sensor but may need to be manually reset if persists !!!')
	    self.sub.unregister()
	    del self.tf_listener
	    del self.tfBuffer
	    self.sub = rospy.Subscriber(self.sensor_topic, self.sensor_msg_type, self.sensor_callback, queue_size=1)
	    self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
	    self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)


class WindSensor:
    def __init__(self):
        self.units = "m/s"
        self.u = []
        self. v = []
        self.pos = []
        self.timestamp = []
        self.u_variance = 0.1 ** 2
        self.v_variance = 0.1 ** 2
        self.covariance = np.diag([self.u_variance, self.v_variance])

    def get_wind(self, map, pos, time=0):
        pos_map_x = round(pos[0] / map.gridsize)
        pos_map_y = round(pos[1] / map.gridsize)
        u_gt = map.udata[pos_map_x, pos_map_y]
        v_gt = map.vdata[pos_map_x, pos_map_y]

        u, v = np.random.multivariate_normal(np.array([u_gt, v_gt]), self.covariance)

        self.u.append(u)
        self.v.append(v)
        self.pos.append([pos_map_x, pos_map_y])
        self.timestamp.append(time)
