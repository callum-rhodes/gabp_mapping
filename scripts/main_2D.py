#!/usr/bin/python

from gabp_mapping import maps, GaBP, sensors
import rospy
import os
from std_msgs.msg import Float32
from nav_msgs.msg import OccupancyGrid
import threading
import numpy as np


def init_ros():
    rospy.init_node('gabp', anonymous=True)
    #  Parameter for defining the filepath to save .npy files
    output_path = rospy.get_param('~output_path', '{}/'.format(os.environ['HOME']))
    #  rospy.loginfo('Saving data to filepath: {}'.format(output_path))

    sensor_topic = rospy.get_param('~sensor_topic', '/gasSensor')
    rospy.loginfo('Using sensor topic: {}'.format(sensor_topic))

    sensor_freq = rospy.get_param('~sensor_freq', '0')
    sensor_freq = float(sensor_freq)
    rospy.loginfo('Expected sensor frequency: {}'.format(sensor_freq))

    map_frame = rospy.get_param('~map_frame', 'map')
    rospy.loginfo('Using map frame: {}'.format(map_frame))

    sensor_child_frame = rospy.get_param('~sensor_child_frame', 'base_link')
    rospy.loginfo('Using sensor child frame: {}'.format(sensor_child_frame))

    map_topic = rospy.get_param('~occ_map_topic', '/map')
    rospy.loginfo('Using map topic: {}'.format(map_topic))

    resolution = rospy.get_param('~map_resolution', 0.2)
    resolution = float(resolution)
    rospy.loginfo('Output map resolution: {}'.format(resolution))

    freq_val = rospy.get_param('~pub_freq', 1)
    freq_val = float(freq_val)
    rospy.loginfo('Topic publish frequency: {}Hz'.format(freq_val))

    use_blank_map = rospy.get_param('~use_blank_map', 'False')
    rospy.loginfo('Using blank map?: {}'.format(use_blank_map))

    if use_blank_map:
        print('IT WORKED')
        rospy.set_param('~occ_freq', 1e-9)
        map_size = rospy.get_param('~map_size', '50,50')
        rospy.loginfo('Setting blank map with size ' + map_size)
    else:
        print('IT DIDNT WORK')
        map_size = []

    oct_update_freq = rospy.get_param('~occ_freq', 0.1)
    oct_update_freq = float(oct_update_freq)
    rospy.loginfo('Occupancy update frequency: {}Hz'.format(oct_update_freq))

    p_val = rospy.get_param('~factor_prior', 0.5)
    p_val = float(p_val)
    o_val = rospy.get_param('~factor_obs', 10)
    o_val = float(o_val)
    t_val = rospy.get_param('~factor_time', 1000000)
    t_val = float(t_val)
    d_val = rospy.get_param('~factor_default', 1e-4)
    d_val = float(d_val)
    eps_val = rospy.get_param('~propagation_epsilon', 1)
    eps_val = float(eps_val)
    metric = rospy.get_param('~distance_metric', 'bhattacharyya')
    rospy.loginfo('Prior precision: %f', p_val)
    rospy.loginfo('Observation precision: %f', o_val)
    rospy.loginfo('Time precision: %f', t_val)
    rospy.loginfo('Default precision: %f', d_val)
    rospy.loginfo('Propagation threshold: %f', eps_val)
    rospy.loginfo('Distance metric: ' + metric)

    return output_path, sensor_topic, sensor_freq, map_frame, sensor_child_frame, map_topic, resolution, freq_val, oct_update_freq, p_val, o_val, t_val, d_val, eps_val, metric, use_blank_map, map_size


if __name__ == '__main__':
    try:
        [output_path, sensor_topic, sensor_freq, map_frame, sensor_child_frame, map_topic, resolution, freq_val, occ_update_freq, p_val, o_val, t_val, d_val, eps_val, metric, blank_bool, map_size] = init_ros()
        sensor_msg_type = Float32
        occ = maps.OccGenerator()
        if blank_bool: # If using the blank map boolean initiates based on the requested size at [0,0]
            map_size = [int(float(x) / resolution) for x in map_size.split(",")] 
            occ.blank2occ(map_size, resolution)
        else: #  Else will read the occupancy grid message for the map
            occ_msg = rospy.wait_for_message(map_topic, OccupancyGrid)
            occ.ros2occ(occ_msg, resolution) #  Converts the occupancy grid msg into a 2D matrix of binary values

        ################ DEFINE GAS SENSORS ###################
        sensor_topic_list = sensor_topic.split(",") #  Extracts the sensor topics
        child_frame_list = sensor_child_frame.split(",") #  Extracts the sensor frame names
        gas_sensor = []
        meas_idx = []
        for sens_num in range(0, len(sensor_topic_list)):
            gas_sensor.append(sensors.GasSensor(1, sensor_topic_list[sens_num], sensor_msg_type, map_frame, child_frame_list[sens_num]))
            meas_idx.append(0)  # Track the measurement id that has already been inserted to graph

        add_measures = sensors.GasSensor(1)  # Dummy sensor to input only new measurements into GaBP

        ################ INITIALISE GABP OBJECT ###################
        tic = rospy.get_rostime()
        GaBP_k = GaBP.G_GaBP_2D(occ, map_frame, lambda_p=p_val, lambda_o=o_val, lambda_d=d_val, epsilon=eps_val, lambda_t=t_val, pub_freq=freq_val, sens_freq=sensor_freq, filepath=output_path)
        toc = rospy.get_rostime()
        rospy.loginfo('------ GaBP initialised ------')
        rospy.loginfo("[Factor Graph Init]: %.3fs", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))

        resid_loop = threading.Thread(target=GaBP_k.residual_loop)  # Start residual propagation in another thread for when not performing wildfire
        resid_loop.start()

        mapLoop_start = rospy.get_time()
	rospy.on_shutdown(GaBP_k.on_shutdown)
	occ_itr = 0
        sim_start = rospy.get_time()

        while not rospy.is_shutdown():
            ################ CHECKING FOR NEW MEASUREMENTS ###################
            for sens_num in range(0, len(sensor_topic_list)):
                data_len = len(gas_sensor[sens_num].data)
                if data_len > meas_idx[sens_num]: # If more measurements in sensor object than in inserted into GaBP
                    for data_idx in range(meas_idx[sens_num], data_len): # Adds only the new measurements into the dummy sensor
                        add_measures.data.append(gas_sensor[sens_num].data[data_idx])
                        add_measures.pose.append(gas_sensor[sens_num].pose[data_idx])
                        add_measures.timestamp.append(gas_sensor[sens_num].timestamp[data_idx])
                    meas_idx[sens_num] = data_len

            ################ ADD NEW MEASUREMENTS TO GABP ###################
            if add_measures.data:
                tic = rospy.get_rostime()
                GaBP_k.add_obs(add_measures)  #  inserts new sensors measurements into solver
                toc = rospy.get_rostime()
                rospy.loginfo("[Wildfire]:%d measurements in %.3fs", len(add_measures.data),  toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                add_measures.data = [] # resets the dummy sensor to blank for the next set of measurements
                add_measures.pose = []
                add_measures.timestamp = []

            if rospy.get_time() - mapLoop_start > 1 / occ_update_freq:
                ## Uncomment if you need to save conc/var maps to file
		#  np.save(output_path + 'GaBP_mean_matrix'+ str(occ_itr) +'.npy', GaBP_k.mean)
                #  rospy.loginfo('mean map saved to: ' + output_path + 'GaBP_mean_matrix'+ str(occ_itr) +'.npy')
                #  np.save(output_path + 'GaBP_var_matrix'+ str(occ_itr) +'.npy', GaBP_k.var)
                #  rospy.loginfo('variance map saved to: ' + output_path + 'GaBP_var_matrix'+ str(occ_itr) +'.npy')
                #  np.save(output_path + 'GaBP_meta_data'+ str(occ_itr) +'.npy', np.array([GaBP_k.origin, GaBP_k.gridsize, GaBP_k.N, len(GaBP_k.z_pos),GaBP_k.msgs_wild, GaBP_k.msgs_resid, rospy.get_time() - sim_start]))
                #  rospy.loginfo('meta data saved to: ' + output_path + 'GaBP_meta_data'+ str(occ_itr) +'.npy')
		occ_itr += 1
                rospy.loginfo("+++++ UPDATING OCCUPANCY MAP +++++")
                tic = rospy.get_rostime()
                occ = maps.OccGenerator()
                occ_msg = rospy.wait_for_message(map_topic, OccupancyGrid)
                occ.ros2occ(occ_msg, resolution)
                toc = rospy.get_rostime()
                rospy.loginfo("[Occupancy grid]: converted in %.3fs", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
		tic = rospy.get_rostime()
                GaBP_k.update_occ(occ)
                toc = rospy.get_rostime()
                rospy.loginfo("+++++ FACTOR GRAPH UPDATED in %.3fs +++++", toc.secs - tic.secs + 1e-9 * (toc.nsecs - tic.nsecs))
                mapLoop_start = rospy.get_time()

    except rospy.ROSInterruptException:
        GaBP_k.stop_thread = True
