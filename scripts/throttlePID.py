#!/usr/bin/python2.7
import rospy
from std_msgs.msg import Float32
from std_msgs.msg import Int32

rospy.init_node('PID_throttle', anonymous=True)

out_topic = rospy.get_param('~out_topic', '/integrated')
rospy.loginfo('Ouputting on topic: {}'.format(out_topic))

topic_type = rospy.get_param('~topic_type', 'Int32')
rospy.loginfo('Ouputting on topic: {}'.format(out_topic))

pub_PID = rospy.Publisher(out_topic, Float32, queue_size=1)

pub_freq = rospy.get_param('~publish_freq', 2)
rospy.loginfo('Integrating PID sensor with freq: {}'.format(pub_freq))

sensor_topic = rospy.get_param('~PID_topic', '/GasSensor1')
rospy.loginfo('Getting PID data from topic: {}'.format(sensor_topic))

class Integrator:
    def __init__(self, pub_freq=0.5, sensor_topic='/GasSensor1', topic_type=Int32):
        self.input_topic = sensor_topic
        self.integration_period = 1. / pub_freq

        self.data_store = [0]
        self.time_store = [rospy.get_time()]
	if topic_type == 'Float32':
		rospy.Subscriber(sensor_topic, Float32, self.callback)
	else:
        	rospy.Subscriber(sensor_topic, Int32, self.callback)

    def callback(self, data):
        self.data_store.append(data.data)
        self.time_store.append(rospy.get_time())

    def getIntegratedValue(self):
        t_now = rospy.get_time()
        t_delta = 0.
        i = 1
        summer = 0.
        while t_delta < self.integration_period and i != len(self.data_store):
            summer += self.data_store[-i]
            t_delta = t_now - self.time_store[-i]
            i += 1

        val = float(summer) / i
        trunc = len(self.data_store) - i
        del self.data_store[:trunc]
        del self.time_store[:trunc]

        return val


def republish_PID():
    rospy.sleep(0.5)
    PID_msg = Float32()
    PID_int = Integrator(pub_freq, sensor_topic, topic_type)
    rate = rospy.Rate(pub_freq)
    while not rospy.is_shutdown():
        PID_value = PID_int.getIntegratedValue()
        PID_msg.data = PID_value
        pub_PID.publish(PID_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        republish_PID()
    except rospy:
        pass
