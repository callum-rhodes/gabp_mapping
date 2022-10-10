# Gaussian belief propagation for 2D and 3D gas distribution mapping

This package contains ROS integrated software for performing 2D and 3D gas distribution with a mobile sensor. A youtube video of the 3D mapping system being demonstrated onboard a mobile platform can be found [here](https://youtu.be/crAJd4afW8c). If you wish to cite our work please use:
> Rhodes, C., Liu, C., & Chen, W. H. (2022, October). Scalable probabilistic gas distribution mapping using Gaussian belief propagation. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE.

![GaBP applied to a gas mapping scenario](https://github.com/callum-rhodes/gabp_mapping/blob/main/githubGaBP.png?raw=true)

## Software Requirements
### System:
- [ROS Melodic](http://wiki.ros.org/melodic) (tested and deployed on Melodic)
- Octomap_server
    - ```sudo apt install ros-melodic-octomap-server```
- Python 2.7

### Python Packages:
- Numpy
- octomap-python
    - ```pip install octomap-python```
    - [GitHub repository for issues with installation](https://github.com/wkentaro/octomap-python)
    - Specific issues with _ZN7octomap14AbstractOcTree4readERKSs see [here](https://github.com/wkentaro/octomap-python/issues/5) and [here](https://github.com/wkentaro/octomap-python/issues/3)
    - Specific issues with libdynamicedt3d see [here](https://github.com/wkentaro/octomap-python/issues/1)

## Installation
1) Change directory to your catkin workspace source folder, e.g.
    - ```cd ~/catkin_ws/src```
1) Clone the repository
    - ```git clone https://github.com/callum-rhodes/gabp_mapping.git```
1) Change to your catkin folder
    - ```cd ..```
1) Catkin make the package
    - ```catkin_make --only-pkg-with-deps gabp_mapping```


## Running the package
### 2D mapping
When running a 2D node, the "main_2D.py" node should be used. This node subscribes to a nav_msgs/OccupancyGrid message to build the factor graph. If an occupancy map is not available, the "use_blank_map" parameter can be set to 'True' and a corresponding size set on the "map_size" parameter e.g. '50,50' (blank maps always start with an origin of [0,0]).

An example 2D launch file is included in /launch/Example_2D.launch.

### 3D mapping
When running a 3D node, the "main_3D.py" node should be used. This node subscribes to the octomap_msgs/Octomap binary message topic and uses the map saving feature of octomap_server to save the tree in the .bt format that octomap-python requires. to build the factor graph. If an binary octomap is not available, the "use_blank_map" parameter can be set to 'True' and a corresponding size set on the "map_size" parameter e.g. '50,50,10' (blank maps always start with an origin of [0,0,0]).

An example 3D launch file is included in /launch/Example_3D.launch.

### Visualisation

An example .rviz config file can be found in /rviz/gabp_3D.rviz. This can be used for both 2D and 3D nodes. For 2D mapping, images show both the marginal mean map and the variance map (on /gabp/mean/img and /gabp/var/img repectively) and a marker shows the marginal mean map projected onto the scenario floor (/gabp/mean/marker).
For 3D mapping, images show both the projected marginal mean map and the projected variance map (on /gabp/mean/img and /gabp/var/img repectively), where the projection is averaged along the z-axis. A marker shows the marginal mean map in 3D space with decreasing alpha for lower concentrations (/gabp/mean/marker).

## Example bag files

An example dataset for running both 2D and 3D gas mapping can be found [here](https://drive.google.com/file/d/1-fEBbk5spXurhdF5mjQEKchi83KfWyIC/view?usp=sharing). To run the dataset follow these instructions:

1. Download the .tar file and extract using ```tar -xvf Example_bags.tar.gz``` 
2. Start the ros master using ```roscore``` 
3. Open a new terminal and set ```rosparam set use_sim_time true``` 
4. ```cd``` to the extracted bag files location 
5. Begin the rosbag using ```rosbag play --clock --pause *.bag``` 
6. Open a new terminal and enter ```roscd gabp_mapping/rviz``` 
7. Begin RViz using ```rviz -d gabp_3D.rviz``` 
8. Open a new terminal and run ```roslaunch gabp_mapping Example_3D.launch``` , or ```roslaunch gabp_mapping Example_2D.launch``` 
9. Go back to the terminal running the rosbag, then hit <kbd>Space</kbd>
10. RViz should now populate showing the Octomap and gas distribution markers as the robot explores its environment

## Published topics

- /gabp/mean/img ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)) \
    Marginal mean value of each factor in the factor graph plotted as a colourised image

- /gabp/mean/marker ([visualization_msgs/Marker](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html)) \
    Marginal mean value of each factor in the factor graph plotted in space as a colourised voxel
    
- /gabp/mean/matrix ([std_msgs/Float32MultiArray](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32MultiArray.html)) \
    Marginal mean values published as a multi-array matrix. Layout of matrix defined in /gabp/mean/matrix/layout.  

- /gabp/var/img ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)) \
    Marginal variance value of each factor in the factor graph plotted as a colourised image
    
- /gabp/var/matrix ([std_msgs/Float32MultiArray](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32MultiArray.html)) \
    Marginal variance values published as a multi-array matrix. Layout of matrix defined in /gabp/var/matrix/layout.  

- /gabp/state (gabp_mapping/state) \
  State message of the factor graph: 
    * num_nodes - Number of nodes in the factor graph 
    * num_measurements - Number of measurements 
    * msgs_wild - Number of messages passed during wildfire iterations 
    * msgs_residual - Number of messages passed during residual iterations 
    * msgs_total - Total number of messages passed 
    * max_mean - Maximum mean marginal value 
    * max_var - Maximum variance value 
    * max_measurement - Maximum measured value inserted into the graph
        
## Parameters

~sensor_topic (str, default: /gasSensor)
> Topic from which sensor measurements are taken, multiple sensors can be used via comma seperation e.g. '/gas1,/gas2,/gas3' (topic type should be Float32)

~sensor_freq (float, default: 0)
> Expected sensor frequency of incoming messages. This limits the maximum time per wildfire iteration to match incoming data rate. If set to 0, each wildfire iteration will run until residuals reach propagation threshold set on ~epsilon_threshold. Sensor messages received whilst iterating are stored and resolved in the next iteration. 

~sensor_child_frame (str, default: base_link)
> TF from which the sensor pose is taken for inserting sensor measurement into the map. If using multiple sensors then include all tf's via comma serperation e.g. '/gas1base,/gas2base,/gas3base'.

~map_frame (str, default: map)
> Parent TF of sensor_child_frame to link sensor to map frame e.g. map --> base_link

~occ_map_topic (str, default: /map)
> Topic that defines the occupancy grid / Octomap topic to read from

~occ_freq (float, default: 0.1)
> Frequency at which the occupancy/octomap is updated to check for new prior factors and neighbours

~pub_freq (float, default: 1)
> Target frequency for publishing the image, marker and state messages

~map_resolution (float, default: 0.3)
> Desired resolution of the factor graph

~use_blank_map (str, default: False)
> If set to 'True' a blank map (no obstacles) is used to define the mapping domain. Size is set by the ~map_size parameter.

~map_size (str, default: 50,50,20)
> The size of the blank map in x,y,z. If using the 2D solver, 2 arguments should be given e.g. '50,50'

~factor_prior (float, default: 0.5)
> Prior factor precision value between nodes in the factor graph

~factor_obs (float, default: 10)
> Observation factor precision value of measurements to their corresponding node

~factor_time (float, default: 1e10)
> Decay factor precision value that increases measurement uncertainty over time. Lower means measurements decay quicker

~factor_default (float, default: 1e-4)
> Default factor precision value of initial cell concentration of zero

~propagation_threshold (float, default: 1)
> Threshold between consecutive messages along an edge that defines wildfire propagation and factor graph growth

~distance_metric (str, default: Bhattacharyya)
> Metric of the distance between consecutive messages that effects wildfire propagation, factor growth and the residual queue. Available options 'Bhattacharyya', 'Mahalanobis' and 'Euclidean'.
