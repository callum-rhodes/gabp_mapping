#!/usr/bin/python

from gabp_mapping import maps, GaBP, sensors
import rospy
import octomap
from std_msgs.msg import Float32
from octomap_msgs.msg import Octomap
import threading
import os


def init_ros():
    rospy.init_node('gabp', anonymous=True)

    #  rospy.set_param('~pose_topic', '/mavros/local_position/pose')
    pose_topic = rospy.get_param('~pose_topic', '/pose')
    rospy.loginfo('Using pose topic: {}'.format(pose_topic))

    #  rospy.set_param('~sensor_topic', '/sensorLow/integrated,/sensorHigh/integrated')
    #  rospy.set_param('~sensor_topic', '/PID')
    sensor_topic = rospy.get_param('~sensor_topic', '/gasSensor')
    rospy.loginfo('Using sensor topic: {}'.format(sensor_topic))

    #  rospy.set_param('~map_frame', '/husky/map')
    map_frame = rospy.get_param('~map_frame', '/map')
    rospy.loginfo('Using map frame: {}'.format(map_frame))

    #  rospy.set_param('~sensor_child_frame', '/sensorLow,/sensorHigh')
    #  rospy.set_param('~sensor_child_frame', '/pose')
    sensor_child_frame = rospy.get_param('~sensor_child_frame', '/base_link')
    rospy.loginfo('Using sensor child frame: {}'.format(sensor_child_frame))

    #  rospy.set_param('~occ_map_topic', '/octomap_binary')
    map_topic = rospy.get_param('~occ_map_topic', '/map')
    rospy.loginfo('Using map topic: {}'.format(map_topic))

    resolution = rospy.get_param('~map_resolution', 0.3)
    resolution = float(resolution)
    rospy.loginfo('Output map resolution: {}'.format(resolution))

    freq_val = rospy.get_param('~pub_freq', 1)
    freq_val = float(freq_val)
    rospy.loginfo('Topic publish frequency: {}Hz'.format(freq_val))

    #  rospy.set_param('~use_blank_map', 'True')
    use_blank_map = rospy.get_param('~use_blank_map', 'False')
    rospy.loginfo('Using blank map?: {}'.format(use_blank_map))

    if use_blank_map == 'True':
        rospy.set_param('~occ_freq', 1e-9)
        map_size = rospy.get_param('~map_size', '50,50,20')
        rospy.loginfo('Setting blank map with size ' + map_size)
    else:
        map_size = []

    oct_update_freq = rospy.get_param('~occ_freq', 0.1)
    oct_update_freq = float(oct_update_freq)
    rospy.loginfo('Octomap update frequency: {}Hz'.format(oct_update_freq))

    p_val = rospy.get_param('~factor_prior', 0.5)
    p_val = float(p_val)
    o_val = rospy.get_param('~factor_obs', 10)
    o_val = float(o_val)
    t_val = rospy.get_param('~factor_time', 1000000)
    t_val = float(t_val)
    d_val = rospy.get_param('~factor_default', 1e-4)
    d_val = float(d_val)
    eps_val = rospy.get_param('~propagation_epsilon', 1e1)
    eps_val = float(eps_val)
    #  rospy.set_param('~distance_metric', 'euclidean')
    #  rospy.set_param('~distance_metric', 'mahalanobis')
    metric = rospy.get_param('~distance_metric', 'bhattacharyya')
    rospy.loginfo('Prior precision: %f', p_val)
    rospy.loginfo('Observation precision: %f', o_val)
    rospy.loginfo('Time precision: %f', t_val)
    rospy.loginfo('Default precision: %f', d_val)
    rospy.loginfo('Propagation threshold: %f', eps_val)
    rospy.loginfo('Distance metric: ' + metric)

    return pose_topic, sensor_topic, map_frame, sensor_child_frame, map_topic, resolution, freq_val, oct_update_freq, p_val, o_val, t_val, d_val, eps_val, metric, use_blank_map, map_size


if __name__ == '__main__':
    try:
        [pose_topic, sensor_topic, map_frame, sensor_child_frame, map_topic, resolution, freq_val, oct_update_freq, p_val, o_val, t_val, d_val, eps_val, metric, blank_bool, map_size] = init_ros()
        sensor_msg_type = Float32
        occ = maps.OccGenerator()
        if blank_bool == 'True':
            map_size = [int(float(x) / resolution) for x in map_size.split(",")]
            occ.blank2occ(map_size, resolution)
        else:
            rospy.loginfo('---- Reading octomap message ----')
            # octree = octomap.OcTree('octomap_example/GW3_-1-3_0.05.bt')
            rospy.wait_for_message(map_topic, Octomap)
            os.system('rosrun octomap_server octomap_saver GaBP_oct.bt')
            octree = octomap.OcTree('GaBP_oct.bt')
            occ.oct2occ(octree, resolution)

        ################ DEFINE GAS SENSORS ###################
        sensor_topic_list = sensor_topic.split(",")
        child_frame_list = sensor_child_frame.split(",")
        gas_sensor = []
        meas_idx = []
        for sens_num in range(0, len(sensor_topic_list)):
            gas_sensor.append(sensors.GasSensor(1, sensor_topic_list[sens_num], sensor_msg_type, map_frame, child_frame_list[sens_num]))
            meas_idx.append(0)  # Track the measurement id that has already been inserted to graph

        add_measures = sensors.GasSensor(1)  # Dummy sensor to input only new measurements into GaBP

        ################ INITIALISE GABP OBJECT ###################
        tic = rospy.get_rostime()
        GaBP_k = GaBP.G_GaBP_3D(occ, map_frame, lambda_p=p_val, lambda_o=o_val, lambda_d=d_val, epsilon=eps_val, lambda_t=t_val, distance=metric, pub_freq=freq_val)
        toc = rospy.get_rostime()
        rospy.loginfo('------ GaBP initialised ------')
        rospy.loginfo("[Factor Graph Init]: %.3fs", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))

        resid_loop = threading.Thread(target=GaBP_k.residual_loop)  # Start residual looping
        resid_loop.start()

        octLoop_start = rospy.get_time()

        while not rospy.is_shutdown():
            ################ CHECKING FOR NEW MEASUREMENTS ###################
            for sens_num in range(0, len(sensor_topic_list)):
                data_len = len(gas_sensor[sens_num].data)
                if data_len > meas_idx[sens_num]:
                    for data_idx in range(meas_idx[sens_num], data_len):
                        add_measures.data.append(gas_sensor[sens_num].data[data_idx])
                        add_measures.pose.append(gas_sensor[sens_num].pose[data_idx])
                        add_measures.timestamp.append(gas_sensor[sens_num].timestamp[data_idx])
                    meas_idx[sens_num] = data_len

            ################ ADD NEW MEASUREMENTS TO GABP ###################
            if add_measures.data:
                tic = rospy.get_rostime()
                GaBP_k.add_obs(add_measures)
                toc = rospy.get_rostime()
                rospy.loginfo("[Wildfire]:%d measurements in %.3fs", len(add_measures.data),  toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                add_measures.data = []
                add_measures.pose = []
                add_measures.timestamp = []

            if rospy.get_time() - octLoop_start > 1 / oct_update_freq:
                rospy.loginfo("+++++ UPDATING OCTOMAP +++++")
                tic = rospy.get_rostime()
                os.system('rosrun octomap_server octomap_saver GaBP_oct.bt')
                octree = octomap.OcTree('GaBP_oct.bt')
                toc = rospy.get_rostime()
                rospy.loginfo("[Octomap]: read in %.3fs", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                tic = rospy.get_rostime()
                occ.oct2occ(octree, resolution)
                toc = rospy.get_rostime()
                rospy.loginfo("[Octomap]: converted in %.3fs", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                tic = rospy.get_rostime()
                GaBP_k.update_occ(occ)
                toc = rospy.get_rostime()
                rospy.loginfo("+++++ FACTOR GRAPH UPDATED in %.3fs +++++", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                octLoop_start = rospy.get_time()

    except rospy.ROSInterruptException:
        pass
