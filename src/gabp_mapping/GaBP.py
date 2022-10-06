import numpy as np
import matplotlib.pyplot as plt
import math

import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from gabp_mapping.msg import gabp_state

class G_GaBP_2D:
    def __init__(self, occ, map_frame='/map', lambda_p=0.5, lambda_o=10.0, lambda_d=1e-4, lambda_t=1e6, epsilon=1e-4, distance='euclidean', pub_freq=1, sens_freq=0, filepath='~/'):
        self.time_init = []
        self.time_add = []
        self.time_solve = []
        self.cmap = plt.get_cmap("viridis")
        self.filepath = filepath

        self.frame = map_frame
        self.meanImgPub = rospy.Publisher('gabp/mean/img', Image, queue_size=1)
        self.meanMarkerPub = rospy.Publisher('gabp/mean/marker', Marker, queue_size=1)
        self.meanMatPub = rospy.Publisher('gabp/mean/matrix', Float32MultiArray, queue_size=1)
        rospy.loginfo('Publishing mean map on topic: {}'.format('/gabp/mean'))
        self.varImgPub = rospy.Publisher('/gabp/var/img', Image, queue_size=1)
        self.varMatPub = rospy.Publisher('gabp/var/matrix', Float32MultiArray, queue_size=1)
        rospy.loginfo('Publishing uncertainty map on topic: {}'.format('/gabp/var'))
        self.statePub = rospy.Publisher('gabp/state', gabp_state, queue_size=1)
        rospy.loginfo('Publishing state on topic: {}'.format('/gabp/state'))
        self.rate = rospy.Duration(pub_freq)
        self.wild_bool = True  # Boolean for checking which schedule is being currently being used
        self.updating_bool = False # Boolean for checking if occupancy grid is being updated
        self.stop_thread = False
        self.markerArray = MarkerArray()
        self.meanMatrix = Float32MultiArray()
        self.meanMatrix.layout.dim.append(MultiArrayDimension())
        self.meanMatrix.layout.dim.append(MultiArrayDimension())
        self.varMatrix = Float32MultiArray()
        self.varMatrix.layout.dim.append(MultiArrayDimension())
        self.varMatrix.layout.dim.append(MultiArrayDimension())
        # Check the paper for definition of these parameters below
        self.lambda_p = float(lambda_p)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.lambda_o = float(lambda_o)
        self.eps = epsilon
        self.distance_metric = distance
        if self.distance_metric == 'euclidean':
            self.dm = 0
        elif self.distance_metric == 'bhattacharyya':
            self.dm = 1
        elif self.distance_metric == 'mahalanobis':
            self.dm = 2
        else:
            rospy.loginfo('!! Distance metric not recognised !! - Defaulting to Euclidean')
            self.dm = 0

        self.xlen = occ.xlen
        self.ylen = occ.ylen
        self.occ = occ.data
        self.map2graph = np.array(np.ones((self.xlen, self.ylen)) * -1).astype(int) # Lookup table for x-y to index in factor graph
        self.graph2map = [] # Lookup table for index in factor graph to x-y position
        self.gridsize = occ.gridsize
        self.origin = G_GaBP_2D.round2grid(np.array(occ.origin), self.gridsize)
        self.width = occ.w
        self.length = occ.l

        self.N = int(0) # Stores the number of variables in the factor graph
        self.H_p = [] # Stores the prior precision values of the factor graph as a list of lists
        self.g = [] # observation vector

        self.Neighbours = [] # Stores neighbour indices of each node, list of lists
        self.P = [] # [Marginal precision, incoming message precisions], order is the same order as neighbour list for incoming precisions
        self.u = [] # [Marginal means, incoming message means], order is the same order as neighbour list for incoming precisions
        self.resid = [] # Residual vector storing residuals on a variable
        self.isexpanded = [] # Binary check for is node has been expanded in the graph
        self.msgs_wild = 0
        self.msgs_resid = 0

        self.z = [] # Measurement value
        self.z_pos = [] # Measurement pose
        self.z_time = [] # Measurement time
        self.z_graph_id = [] # Measurement associate graph index
        self.z_freq = 1 / sens_freq # Measurement frequency

        self.mean = np.zeros((self.xlen, self.ylen))
        self.mean_norm = np.zeros((self.xlen, self.ylen))
        self.var = np.zeros((self.xlen, self.ylen))
        self.var_norm = np.zeros((self.xlen, self.ylen))

        self.marker = Marker()
        self.marker.id = 0
        self.marker.header.stamp = rospy.Time.now()
        self.marker.header.frame_id = self.frame
        self.marker.type = self.marker.CUBE_LIST
        self.marker.action = self.marker.ADD
        self.marker.scale.x = self.gridsize
        self.marker.scale.y = self.gridsize
        self.marker.scale.z = self.gridsize

        self.marker.pose.position.x = 0
        self.marker.pose.position.y = 0
        self.marker.pose.position.z = 0
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z = 0
        self.marker.pose.orientation.w = 1

        self.Pii = [] # Prior precision vector
        self.uii = [] # Prior mean vector

        self.timer = rospy.Timer(self.rate, self.timer_callback)

        np.seterr(divide='ignore')

    @staticmethod
    def round2grid(array, nearest):  # Function for taking continuous space poses and finding the nearest grid point, nearest = a grid size in (m)
        rounded = np.round(array / nearest) * nearest
        return rounded

    def addedge(self, parent_id, child): # Function for adding a new edge to the graph
        if self.map2graph[child[0], child[1]] == -1: # Checks x-y position of child node to see if it already exists. -1 = does not exist
            self.N += 1
            child_id = self.N - 1 # N is not zero indexed so last node is always N-1
            self.map2graph[child[0], child[1]] = child_id
            self.graph2map.append(child)
            self.H_p.append([self.lambda_d]) # Add default factor precision
            self.g.append(0) # initialise observation value 
            self.P.append([0]) # initialise marginal precision list 
            self.Pii.append(self.lambda_d) # initialise prior precision value 
            self.u.append([0]) # initialise marginal mean list 
            self.uii.append(0) # initialise prior mean list 
            self.resid.append(0) # initialise residual value
            self.isexpanded.append(False)
            self.Neighbours.append([]) # initialise neighbour list
        else:
            child_id = self.map2graph[child[0], child[1]] # If its already in the graph find its index
        # Add relevant values to reflect new connection in parent
        self.Neighbours[parent_id].append(child_id)
        self.H_p[parent_id][0] += self.lambda_p
        self.H_p[parent_id].append(-self.lambda_p)
        self.Pii[parent_id] += self.lambda_p
        self.P[parent_id].append(self.lambda_p)
        self.u[parent_id].append(0)
        self.uii[parent_id] = self.g[parent_id] / self.Pii[parent_id]
        # Add relevant values to reflect new connection in child
        self.Neighbours[child_id].append(parent_id)
        self.H_p[child_id][0] += self.lambda_p
        self.H_p[child_id].append(-self.lambda_p)
        self.Pii[child_id] += self.lambda_p
        self.P[child_id].append(self.lambda_p)
        self.u[child_id].append(0)
        self.uii[child_id] = self.g[child_id] / self.Pii[child_id]

    def deledge(self, parent_id, child): # Function for deleting an edge from the graph
        try:
            child_id = self.map2graph[child[0], child[1]]
        except:
            child_id = child
        # Add relevant values to reflect broken connection in parent
        j_id = self.Neighbours[parent_id].index(child_id)
        self.Neighbours[parent_id].pop(j_id)
        self.H_p[parent_id][0] -= self.lambda_p
        self.H_p[parent_id].pop(j_id + 1)
        self.Pii[parent_id] -= self.lambda_p
        self.uii[parent_id] = self.g[parent_id] / self.Pii[parent_id]
        self.P[parent_id].pop(j_id + 1)
        self.u[parent_id].pop(j_id + 1)
        # Add relevant values to reflect broken connection in child
        j2s_id = self.Neighbours[child_id].index(parent_id)
        self.Neighbours[child_id].pop(j2s_id)
        self.H_p[child_id][0] -= self.lambda_p
        self.H_p[child_id].pop(j2s_id + 1)
        self.Pii[child_id] -= self.lambda_p
        self.uii[child_id] = self.g[child_id] / self.Pii[child_id]
        self.P[child_id].pop(j2s_id + 1)
        self.u[child_id].pop(j2s_id + 1)

    def addnode(self, location): # Function for adding a new variable node to the graph
        self.N += 1
        self.Neighbours.append([])
        self.H_p.append([self.lambda_d])
        self.g.append(0)
        self.P.append([0])
        self.Pii.append(self.lambda_d)
        self.u.append([0])
        self.uii.append(0)
        self.resid.append(0)
        self.isexpanded.append(True)
        idx = self.N - 1
        x = location[0]
        y = location[1]
        self.graph2map.append([x, y])
        self.map2graph[x, y] = idx
        if self.occ[x, y] == 0: # Check up, down, left, right for neighbours
            if y + 1 < self.ylen:
                if self.occ[x, y + 1] == 0:
                    self.addedge(idx, [x, y + 1])
            if y - 1 >= 0:
                if self.occ[x, y - 1] == 0:
                    self.addedge(idx, [x, y - 1])
            if x + 1 < self.xlen:
                if self.occ[x + 1, y] == 0:
                    self.addedge(idx, [x + 1, y])
            if x - 1 >= 0:
                if self.occ[x - 1, y] == 0:
                    self.addedge(idx, [x - 1, y])

    def expandnode(self, id): # Function checking neighbour connections for a variable node
        x = self.graph2map[id][0]
        y = self.graph2map[id][1]
        if y + 1 < self.ylen:
            if self.map2graph[x, y + 1] not in self.Neighbours[id]:
                if self.occ[x, y + 1] == 0:
                    self.addedge(id, [x, y + 1])  # check if free then adds
            elif self.occ[x, y + 1] == 1:
                self.deledge(id, [x, y + 1])  # checks if occupied then deletes
        if y - 1 >= 0:
            if self.map2graph[x, y - 1] not in self.Neighbours[id]:
                if self.occ[x, y - 1] == 0:
                    self.addedge(id, [x, y - 1])
            elif self.occ[x, y - 1] == 1:
                self.deledge(id, [x, y - 1])
        if x + 1 < self.xlen:
            if self.map2graph[x + 1, y] not in self.Neighbours[id]:
                if self.occ[x + 1, y] == 0:
                    self.addedge(id, [x + 1, y])
            elif self.occ[x + 1, y] == 1:
                self.deledge(id, [x + 1, y])
        if x - 1 >= 0:
            if self.map2graph[x - 1, y] not in self.Neighbours[id]:
                if self.occ[x - 1, y] == 0:
                    self.addedge(id, [x - 1, y])
            elif self.occ[x - 1, y] == 1:
                self.deledge(id, [x - 1, y])

        self.isexpanded[id] = True

    def residual_loop(self): # Residual belief propagation loop
        np.seterr(divide='ignore')
        while not self.stop_thread:
            if not self.wild_bool and not self.updating_bool: # If not performing wildfire or updating occ
                sender = np.array(self.resid).argmax()
                # Update self belief based on incoming messages
                self.P[sender][0] = self.Pii[sender] + np.sum(self.P[sender][1:])
                self.u[sender][0] = 1 / self.P[sender][0] * (self.Pii[sender] * self.uii[sender] + np.sum(
                    np.array(self.P[sender][1:]) * np.array(self.u[sender][1:])))

                self.resid[sender] = 0 # Reset residual as we will be sending out information
                # Update messages being sent to neighbours based on new belief
                if self.Neighbours[sender]:
                    j_r = 1
                    while j_r < len(self.Neighbours[sender]) + 1:
                        j_id = self.Neighbours[sender][j_r - 1] # index of the neighbour in the senders list
                        j2s_id = self.Neighbours[j_id].index(sender) + 1 # index of the sender in the neighbours list

                        P_k = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k = np.float64(self.u[j_id][j2s_id])
                        # Update incoming message for the neighbour
                        self.P[j_id][j2s_id] = -self.H_p[sender][j_r] ** 2 / (self.P[sender][0] - self.P[sender][j_r])
                        self.u[j_id][j2s_id] = (self.P[sender][0] * self.u[sender][0] - self.P[sender][j_r] *
                                                self.u[sender][j_r]) / self.H_p[sender][j_r]

                        P_k2 = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k2 = np.float64(self.u[j_id][j2s_id])
                        # Check which metric using and apply
                        if self.dm == 0:
                            self.resid[j_id] = max(self.resid[j_id], abs(u_k2 - u_k))
                        elif self.dm == 1:
                            self.resid[j_id] = 0.25 * math.log(0.25 * (P_k2 / P_k + P_k / P_k2 + 2)) + 0.25 * (
                                (u_k - u_k2) ** 2 / (1 / P_k + 1 / P_k2))
                        elif self.dm == 2:
                            self.resid[j_id] = abs(u_k2 - u_k) / P_k

                        j_r += 1
                        self.msgs_resid += 1

    def add_obs(self, measurement): # Function that takes a new measurment(s) from top level script and adds to the graph
        self.wild_bool = True # Stop performing residual
        self.Pii = [self.H_p[i][0] for i in range(0, self.N)] # Get current prior precisions
        self.g = [0] * self.N # set obs vector to zero initially
        new_meas = len(measurement.data) # how many new measurements being added
        for new_z_id in range(0, new_meas):
            z_pos_corr = self.round2grid(np.array(measurement.pose[new_z_id]) - self.origin, self.gridsize)
            z_pos_grid = np.array(z_pos_corr / self.gridsize).astype(int)
            if all(z_pos_grid >= 0) and all(z_pos_grid[0:2] < [self.xlen, self.ylen]): # If new measurement is in bounds
                self.z_pos.append(z_pos_grid[0:2])
                self.z_time.append(measurement.timestamp[new_z_id])
                self.z.append(measurement.data[new_z_id])
                self.z_graph_id.append(-1) # currently unknown graph location
            else:
                rospy.logwarn("!!! measurement out of bounds, skipping measurement !!!")

                new_meas -= 1

        for z_id in range(0, len(self.z)):
            if self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1]] == -1:  # if variable doesnt exist, then add a new node
                self.addnode(self.z_pos[z_id])
            idx = self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1]]
            self.z_graph_id[z_id] = idx
            lambda_o = 1. / ((1. / self.lambda_o) + (1. / self.lambda_t) * (self.z_time[-1] - self.z_time[z_id])) # Calculate time varying precision for observations
            self.g[idx] += lambda_o * (- self.z[z_id]) # Update observation vector
            self.Pii[idx] += lambda_o # Update prior precision
            self.uii[idx] = self.g[idx] / self.Pii[idx] # Update prior mean

        self.wildfire_itr(new_meas) # Do wildfire loop

        self.wild_bool = False

    def update_occ(self, occ): # Function for updating map reference frame and values for a new map, to avoid negative indexing
        self.updating_bool = True
        origin_new = self.round2grid(np.array(occ.origin), self.gridsize)
        delta = np.array((self.origin - origin_new) / self.gridsize).astype(np.uint32)
        if self.xlen != occ.xlen or self.ylen != occ.ylen or any(delta !=0):
            self.xlen = occ.xlen
            self.ylen = occ.ylen

            self.origin = origin_new

            self.width = occ.w
            self.length = occ.l

            self.mean = np.zeros((self.xlen, self.ylen))
            self.mean_norm = np.zeros((self.xlen, self.ylen))
            self.var = np.zeros((self.xlen, self.ylen))
            self.var_norm = np.zeros((self.xlen, self.ylen))

            self.map2graph = np.array(np.ones((self.xlen, self.ylen)) * -1).astype(int)
            for i in range(0, len(self.graph2map)): # Update graph positions for new reference frame
                self.graph2map[i][0] += delta[0]
                self.graph2map[i][1] += delta[1]
                if all(np.array(self.graph2map[i]) < [self.xlen, self.ylen]):
                    self.map2graph[self.graph2map[i][0], self.graph2map[i][1]] = i
                else:
                    for child in self.Neighbours[i]:
                        self.deledge(i, child)

            for z_id in range(0, len(self.z)): # Update measurement positions for new reference frame
                self.z_pos[z_id][0] += delta[0]
                self.z_pos[z_id][1] += delta[1]

        self.occ = occ.data
        self.isexpanded = [False] * self.N

        self.updating_bool = False

    def wildfire_itr(self, num_bombs): # Wildfire propagation - the magic sauce
        msgs = 0
        Q = [] # Message queue
        wild_start = rospy.get_time()
        for firebomb in range(1, num_bombs + 1): # Each measurement is a firebomb
            bomb_pos = self.z_graph_id[-firebomb] # Get position of latest measurement
            Q.append(bomb_pos)
            while Q:
                sender = Q[0]
                Q.pop(0)
                # Update self belief based on incoming messages
                self.P[sender][0] = self.Pii[sender] + np.sum(self.P[sender][1:])
                self.u[sender][0] = 1 / self.P[sender][0] * (self.Pii[sender] * self.uii[sender] + np.sum(
                    np.array(self.P[sender][1:]) * np.array(self.u[sender][1:])))

                self.resid[sender] = 0 # Reset residual as we will be sending out information
                # Update messages being sent to neighbours based on new belief
                if self.Neighbours[sender]:
                    j = 1
                    while j < len(self.Neighbours[sender]) + 1:
                        j_id = self.Neighbours[sender][j - 1] # index of the neighbour in the senders list
                        j2s_id = self.Neighbours[j_id].index(sender) + 1 # index of the sender in the neighbours list

                        P_k = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k = np.float64(self.u[j_id][j2s_id])
                        # Update incoming message for the neighbour
                        self.P[j_id][j2s_id] = -self.H_p[sender][j] ** 2 / (self.P[sender][0] - self.P[sender][j])
                        self.u[j_id][j2s_id] = (self.P[sender][0] * self.u[sender][0] - self.P[sender][j] *
                                                self.u[sender][j]) / self.H_p[sender][j]

                        P_k2 = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k2 = np.float64(self.u[j_id][j2s_id])

                        self.msgs_wild += 1
                        # Check which metric using and apply
                        if self.dm == 0:
                            self.resid[j_id] = max(abs(u_k2 - u_k), self.resid[j_id])
                        elif self.dm == 1:
                            self.resid[j_id] = 0.25 * math.log(0.25 * (P_k2 / P_k + P_k / P_k2 + 2)) + 0.25 * (
                                (u_k - u_k2) ** 2 / (1 / P_k + 1 / P_k2))
                        elif self.dm == 2:
                            self.resid[j_id] = abs(u_k2 - u_k) / P_k
                        # If residual is above threshold, check if neighbour node j has been expanded for expansions nodes e
                        if self.resid[j_id] > self.eps:
                            if self.isexpanded[j_id] is False:
                                self.expandnode(j_id)
                            if j_id not in Q: # If neighbour is not in queue for update then add it
                                Q.append(j_id)
                        else:
                            converged = True # debug
                        j += 1

                if rospy.get_time() - wild_start > self.z_freq: # If wildfire has run for longer than expected sensor update then stop
                    Q = []

        done = True # debug

    def get_means(self): # Get marginal means into matrix form
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                idx = self.map2graph[x, y]
                if idx != -1:
                    self.mean[x, y] = -self.u[idx][0]
        if self.mean.max() > 0:
            self.mean_norm = self.mean / self.mean.max()

    def get_uncertainty(self): # Get marginal precisions into matrix form
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                idx = self.map2graph[x, y]
                if idx != -1:
                    if self.P[idx][0] != 0:
                        self.var[x, y] = min(1 / self.P[idx][0], 1)
        if self.var.max() > 0:
            self.var_norm = self.var / self.var.max()

    def show_means(self, handle=1): # Old plot functions

        fig = plt.figure(handle)
        plt.clf()
        cax = plt.matshow(self.mean, fig, vmin=0, extent=[0, self.xlen, self.ylen, 0], cmap=self.cmap)
        fig.colorbar(cax)
        plt.show(block=False)
        plt.pause(0.001)

    def show_uncertainty(self, handle=2): # Old plot functions

        fig = plt.figure(handle)
        plt.clf()
        cax = plt.matshow(self.uncertainty, fig, vmin=0, vmax=5, extent=[0, self.width, self.height, 0], cmap=self.cmap)
        fig.colorbar(cax)
        plt.show(block=False)
        plt.pause(0.001)

    def publish_meanimage(self): # Publish mean matrix as an image
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = self.ylen
        msg.width = self.xlen
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        if self.mean.max() != 0:
            im_mat = np.array(
                self.cmap(np.transpose(self.mean) / self.mean.max()) * [255, 255, 255, 1], dtype=np.uint8)
        else:
            im_mat = np.zeros((msg.height, msg.width, 4), dtype=np.uint8)
            zero_colour = [255, 255, 255, 1] * np.array(self.cmap(0))
            im_mat[:, :] = zero_colour
        msg.step = 3 * msg.width
        msg.data = np.flipud(im_mat[:, :, 0:3]).tobytes()
        self.meanImgPub.publish(msg)

    def publish_meanmatrix(self): # Publish mean matrix to ROS
        self.meanMatrix.layout.dim[0].label = "x"
        self.meanMatrix.layout.dim[1].label = "y"
        self.meanMatrix.layout.dim[0].size = self.xlen
        self.meanMatrix.layout.dim[1].size = self.ylen
        self.meanMatrix.layout.dim[0].stride = self.xlen * self.ylen
        self.meanMatrix.layout.dim[1].stride = self.ylen
        self.meanMatrix.layout.data_offset = 0
        self.meanMatrix.data = self.mean.flatten()
        self.meanMatPub.publish(self.meanMatrix)

    def publish_meanmarker(self): # Publish mean matrix as an marker for RViz
        self.marker.points = []
        self.marker.colors = []
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                alpha = self.mean_norm[x, y]
                if alpha >= 0.2:
                    point_msg = Point()
                    point_msg.x = x * self.gridsize + self.origin[0]
                    point_msg.y = y * self.gridsize + self.origin[1]
                    point_msg.z = 0
                    point_colour = ColorRGBA()
        	    rgb = self.cmap(alpha)
                    point_colour.r = rgb[0]
                    point_colour.g = rgb[1]
                    point_colour.b = rgb[2]
                    point_colour.a = alpha
                    self.marker.points.append(point_msg)
                    self.marker.colors.append(point_colour)

        self.marker.header.stamp = rospy.Time.now()
        self.meanMarkerPub.publish(self.marker)

    def publish_varimage(self): # Publish variance matrix as an image
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = self.ylen
        msg.width = self.xlen
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        if self.var_norm.max() != 0:
            im_mat = np.array(
                self.cmap(np.transpose(self.var_norm)) * [255, 255, 255, 1],
                dtype=np.uint8)
        else:
            im_mat = np.zeros((msg.height, msg.width, 4), dtype=np.uint8)
            zero_colour = [255, 255, 255, 1] * np.array(self.cmap(0))
            im_mat[:, :] = zero_colour
        msg.step = 3 * msg.width
        msg.data = np.flipud(im_mat[:, :, 0:3]).tobytes()
        self.varImgPub.publish(msg)

    def publish_varmatrix(self): # Publish variance matrix to ROS
        self.varMatrix.layout.dim[0].label = "x"
        self.varMatrix.layout.dim[1].label = "y"
        self.varMatrix.layout.dim[0].size = self.xlen
        self.varMatrix.layout.dim[1].size = self.ylen
        self.varMatrix.layout.dim[0].stride = self.xlen * self.ylen
        self.varMatrix.layout.dim[1].stride = self.ylen
        self.varMatrix.layout.data_offset = 0
        self.varMatrix.data = self.var.flatten()
        self.varMatPub.publish(self.varMatrix)

    def publish_state(self): # Publish GaBP state message
        msg = gabp_state()
        msg.header.stamp = rospy.Time.now()
        msg.num_nodes = self.N
        msg.num_measurements = len(self.z_pos)
        msg.msgs_wild = self.msgs_wild
        msg.msgs_residual = self.msgs_resid
        msg.msgs_total = self.msgs_resid + self.msgs_wild
        msg.max_mean = self.mean.max()
        msg.max_var = self.var.max()
        msg.max_measurement = max(self.z)
        self.statePub.publish(msg)

    def timer_callback(self, timer): # Callback for publishing ROS data
        try:
            if self.updating_bool is False: # Dont get means if updating graph positions
                self.get_means()
            if self.updating_bool is False:
                self.get_uncertainty()
            #if self.mean.max() > 0:
            self.publish_meanimage()
            self.publish_varimage()
            self.publish_meanmarker()
            self.publish_meanmatrix()
            self.publish_varmatrix()
            self.publish_state()
        except:
            rospy.logwarn("!!! Error publishing ROS data, will retry !!!")

    def on_shutdown(self):
        self.stop_thread = True


class G_GaBP_3D: # 3D GaBP class, everything is broadly the same except considers z-axis connections and produces a 2.5D mean and variance ROS image instead of 2D
    def __init__(self, occ, map_frame='/map', lambda_p=0.5, lambda_o=10.0, lambda_d=1e-4, lambda_t=1e6, epsilon=1e-4, distance='euclidean', pub_freq=1, sens_freq=0, filepath='~/'):
        self.time_init = []
        self.time_add = []
        self.time_solve = []
        self.cmap = plt.get_cmap("viridis")
        self.filepath = filepath

        self.frame = map_frame
        self.meanImgPub = rospy.Publisher('gabp/mean/img', Image, queue_size=1)
        self.meanMarkerPub = rospy.Publisher('gabp/mean/marker', Marker, queue_size=1)
        self.meanMatPub = rospy.Publisher('gabp/mean/matrix', Float32MultiArray, queue_size=1)
        rospy.loginfo('Publishing mean map on topic: {}'.format('/gabp/mean'))
        self.varImgPub = rospy.Publisher('/gabp/var/img', Image, queue_size=1)
        self.varMatPub = rospy.Publisher('gabp/var/matrix', Float32MultiArray, queue_size=1)
        rospy.loginfo('Publishing uncertainty map on topic: {}'.format('/gabp/var'))
        self.statePub = rospy.Publisher('gabp/state', gabp_state, queue_size=1)
        rospy.loginfo('Publishing state on topic: {}'.format('/gabp/state'))
        self.rate = rospy.Duration(pub_freq)
        self.wild_bool = True
        self.updating_bool = False
        self.stop_thread = False
        self.markerArray = MarkerArray()
        self.meanMatrix = Float32MultiArray()
        self.meanMatrix.layout.dim.append(MultiArrayDimension())
        self.meanMatrix.layout.dim.append(MultiArrayDimension())
        self.meanMatrix.layout.dim.append(MultiArrayDimension())
        self.varMatrix = Float32MultiArray()
        self.varMatrix.layout.dim.append(MultiArrayDimension())
        self.varMatrix.layout.dim.append(MultiArrayDimension())
        self.varMatrix.layout.dim.append(MultiArrayDimension())

        self.lambda_p = float(lambda_p)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.lambda_o = float(lambda_o)
        self.eps = epsilon
        self.distance_metric = distance
        if self.distance_metric == 'euclidean':
            self.dm = 0
        elif self.distance_metric == 'bhattacharyya':
            self.dm = 1
        elif self.distance_metric == 'mahalanobis':
            self.dm = 2
        else:
            rospy.loginfo('!! Distance metric not recognised !! - Defaulting to Euclidean')
            self.dm = 0

        self.xlen = occ.xlen
        self.ylen = occ.ylen
        self.zlen = occ.zlen
        self.occ = occ.data
        self.map2graph = np.array(np.ones((self.xlen, self.ylen, self.zlen)) * -1).astype(int)
        self.graph2map = []
        self.gridsize = occ.gridsize
        self.origin = G_GaBP_3D.round2grid(np.array(occ.origin), self.gridsize)
        self.width = occ.w
        self.length = occ.l
        self.height = occ.h

        self.N = int(0)
        self.H_p = []
        self.g = []

        self.Neighbours = []
        self.P = []
        self.u = []
        self.resid = []
        self.isexpanded = []
        self.msgs_resid = 0
        self.msgs_wild = 0

        self.z = []
        self.z_pos = []
        self.z_time = []
        self.z_graph_id = []
        self.z_freq = 1 / sens_freq

        self.mean = np.zeros((self.xlen, self.ylen, self.zlen))
        self.mean_norm = np.zeros((self.xlen, self.ylen, self.zlen))
        self.mean_projected = np.zeros((self.xlen, self.ylen))
        self.var = np.zeros((self.xlen, self.ylen, self.zlen))
        self.var_norm = np.zeros((self.xlen, self.ylen, self.zlen))
        self.var_projected = np.zeros((self.xlen, self.ylen))

        self.marker = Marker()
        self.marker.id = 0
        self.marker.header.stamp = rospy.Time.now()
        self.marker.header.frame_id = self.frame
        self.marker.type = self.marker.CUBE_LIST
        self.marker.action = self.marker.ADD
        self.marker.scale.x = self.gridsize
        self.marker.scale.y = self.gridsize
        self.marker.scale.z = self.gridsize

        self.marker.pose.position.x = 0
        self.marker.pose.position.y = 0
        self.marker.pose.position.z = 0
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z = 0
        self.marker.pose.orientation.w = 1

        self.Pii = []
        self.uii = []

        self.timer = rospy.Timer(self.rate, self.timer_callback)

        np.seterr(divide='ignore')

    @staticmethod
    def round2grid(array, nearest):
        rounded = np.round(array / nearest) * nearest
        return rounded

    def addedge(self, parent_id, child):
        if self.map2graph[child[0], child[1], child[2]] == -1:
            self.N += 1
            child_id = self.N - 1
            self.map2graph[child[0], child[1], child[2]] = child_id
            self.graph2map.append(child)
            self.H_p.append([self.lambda_d])
            self.g.append(0)
            self.P.append([0])
            self.Pii.append(self.lambda_d)
            self.u.append([0])
            self.uii.append(0)
            self.resid.append(0)
            self.isexpanded.append(False)
            self.Neighbours.append([])
        else:
            child_id = self.map2graph[child[0], child[1], child[2]]
        self.Neighbours[parent_id].append(child_id)
        self.H_p[parent_id][0] += self.lambda_p
        self.H_p[parent_id].append(-self.lambda_p)
        self.Pii[parent_id] += self.lambda_p
        self.P[parent_id].append(self.lambda_p)
        self.u[parent_id].append(0)
        self.uii[parent_id] = self.g[parent_id] / self.Pii[parent_id]

        self.Neighbours[child_id].append(parent_id)
        self.H_p[child_id][0] += self.lambda_p
        self.H_p[child_id].append(-self.lambda_p)
        self.Pii[child_id] += self.lambda_p
        self.P[child_id].append(self.lambda_p)
        self.u[child_id].append(0)
        self.uii[child_id] = self.g[child_id] / self.Pii[child_id]

    def deledge(self, parent_id, child):
        try:
            child_id = self.map2graph[child[0], child[1], child[2]]
        except:
            child_id = child

        j_id = self.Neighbours[parent_id].index(child_id)
        self.Neighbours[parent_id].pop(j_id)
        self.H_p[parent_id][0] -= self.lambda_p
        self.H_p[parent_id].pop(j_id + 1)
        self.Pii[parent_id] -= self.lambda_p
        self.uii[parent_id] = self.g[parent_id] / self.Pii[parent_id]
        self.P[parent_id].pop(j_id + 1)
        self.u[parent_id].pop(j_id + 1)

        j2s_id = self.Neighbours[child_id].index(parent_id)
        self.Neighbours[child_id].pop(j2s_id)
        self.H_p[child_id][0] -= self.lambda_p
        self.H_p[child_id].pop(j2s_id + 1)
        self.Pii[child_id] -= self.lambda_p
        self.uii[child_id] = self.g[child_id] / self.Pii[child_id]
        self.P[child_id].pop(j2s_id + 1)
        self.u[child_id].pop(j2s_id + 1)

    def addnode(self, location):
        self.N += 1
        self.Neighbours.append([])
        self.H_p.append([self.lambda_d])
        self.g.append(0)
        self.P.append([0])
        self.Pii.append(self.lambda_d)
        self.u.append([0])
        self.uii.append(0)
        self.resid.append(0)
        self.isexpanded.append(True)
        idx = self.N - 1
        x = location[0]
        y = location[1]
        z = location[2]
        self.graph2map.append([x, y, z])
        self.map2graph[x, y, z] = idx
        if self.occ[x, y, z] == 0:
            if y + 1 < self.ylen:
                if self.occ[x, y + 1, z] == 0:
                    self.addedge(idx, [x, y + 1, z])
            if y - 1 >= 0:
                if self.occ[x, y - 1, z] == 0:
                    self.addedge(idx, [x, y - 1, z])
            if x + 1 < self.xlen:
                if self.occ[x + 1, y, z] == 0:
                    self.addedge(idx, [x + 1, y, z])
            if x - 1 >= 0:
                if self.occ[x - 1, y, z] == 0:
                    self.addedge(idx, [x - 1, y, z])
            if z + 1 < self.zlen:
                if self.occ[x, y, z + 1] == 0:
                    self.addedge(idx, [x, y, z + 1])
            if z - 1 >= 0:
                if self.occ[x, y, z - 1] == 0:
                    self.addedge(idx, [x, y, z - 1])

    def expandnode(self, id):
        x = self.graph2map[id][0]
        y = self.graph2map[id][1]
        z = self.graph2map[id][2]
        if y + 1 < self.ylen:
            if self.map2graph[x, y + 1, z] not in self.Neighbours[id]:
                if self.occ[x, y + 1, z] == 0:
                    self.addedge(id, [x, y + 1, z])  # check if free then adds
            elif self.occ[x, y + 1, z] == 1:
                self.deledge(id, [x, y + 1, z])  # checks if occupied then deletes
        if y - 1 >= 0:
            if self.map2graph[x, y - 1, z] not in self.Neighbours[id]:
                if self.occ[x, y - 1, z] == 0:
                    self.addedge(id, [x, y - 1, z])
            elif self.occ[x, y - 1, z] == 1:
                self.deledge(id, [x, y - 1, z])
        if x + 1 < self.xlen:
            if self.map2graph[x + 1, y, z] not in self.Neighbours[id]:
                if self.occ[x + 1, y, z] == 0:
                    self.addedge(id, [x + 1, y, z])
            elif self.occ[x + 1, y, z] == 1:
                self.deledge(id, [x + 1, y, z])
        if x - 1 >= 0:
            if self.map2graph[x - 1, y, z] not in self.Neighbours[id]:
                if self.occ[x - 1, y, z] == 0:
                    self.addedge(id, [x - 1, y, z])
            elif self.occ[x - 1, y, z] == 1:
                self.deledge(id, [x - 1, y, z])
        if z + 1 < self.zlen:
            if self.map2graph[x, y, z + 1] not in self.Neighbours[id]:
                if self.occ[x, y, z + 1] == 0:
                    self.addedge(id, [x, y, z + 1])
            elif self.occ[x, y, z + 1] == 1:
                self.deledge(id, [x, y, z + 1])
        if z - 1 >= 0:
            if self.map2graph[x, y, z - 1] not in self.Neighbours[id]:
                if self.occ[x, y, z - 1] == 0:
                    self.addedge(id, [x, y, z - 1])
            elif self.occ[x, y, z - 1] == 1:
                self.deledge(id, [x, y, z - 1])

        self.isexpanded[id] = True

    def residual_loop(self):
        np.seterr(divide='ignore')
        while not self.stop_thread:
            if not self.wild_bool and not self.updating_bool:
                sender = np.array(self.resid).argmax()

                self.P[sender][0] = self.Pii[sender] + np.sum(self.P[sender][1:])
                self.u[sender][0] = 1 / self.P[sender][0] * (self.Pii[sender] * self.uii[sender] + np.sum(
                    np.array(self.P[sender][1:]) * np.array(self.u[sender][1:])))

                self.resid[sender] = 0

                if self.Neighbours[sender]:
                    j_r = 1
                    while j_r < len(self.Neighbours[sender]) + 1:
                        j_id = self.Neighbours[sender][j_r - 1]
                        j2s_id = self.Neighbours[j_id].index(sender) + 1

                        P_k = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k = np.float64(self.u[j_id][j2s_id])

                        self.P[j_id][j2s_id] = -self.H_p[sender][j_r] ** 2 / (self.P[sender][0] - self.P[sender][j_r])
                        self.u[j_id][j2s_id] = (self.P[sender][0] * self.u[sender][0] - self.P[sender][j_r] *
                                                self.u[sender][j_r]) / self.H_p[sender][j_r]

                        P_k2 = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k2 = np.float64(self.u[j_id][j2s_id])

                        if self.dm == 0:
                            self.resid[j_id] = max(self.resid[j_id], abs(u_k2 - u_k))
                        elif self.dm == 1:
                            self.resid[j_id] = 0.25 * math.log(0.25 * (P_k2 / P_k + P_k / P_k2 + 2)) + 0.25 * (
                                (u_k - u_k2) ** 2 / (1 / P_k + 1 / P_k2))
                        elif self.dm == 2:
                            self.resid[j_id] = abs(u_k2 - u_k) / P_k

                        j_r += 1
                        self.msgs_resid += 1

    def add_obs(self, measurement):
        self.wild_bool = True
        self.Pii = [self.H_p[i][0] for i in range(0, self.N)]
        self.g = [0] * self.N
        new_meas = len(measurement.data)
        for new_z_id in range(0, new_meas):
            z_pos_corr = self.round2grid(np.array(measurement.pose[new_z_id]) - self.origin, self.gridsize)
            z_pos_grid = np.array(z_pos_corr / self.gridsize).astype(int)
            if all(z_pos_grid >= 0) and all(z_pos_grid[0:3] < [self.xlen, self.ylen, self.zlen]):
                self.z_pos.append(z_pos_grid[0:3])
                self.z_time.append(measurement.timestamp[new_z_id])
                self.z.append(measurement.data[new_z_id])
                self.z_graph_id.append(-1)
            else:
                rospy.logwarn("!!! measurement out of bounds, skipping measurement !!!")
                new_meas -= 1

        for z_id in range(0, len(self.z)):
            if self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1], self.z_pos[z_id][2]] == -1:  # cell factor already exists
                self.addnode(self.z_pos[z_id])
            idx = self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1], self.z_pos[z_id][2]]
            self.z_graph_id[z_id] = idx
            lambda_o = 1. / ((1. / self.lambda_o) + (1. / self.lambda_t) * (self.z_time[-1] - self.z_time[z_id]))
            self.g[idx] += lambda_o * (- self.z[z_id])
            self.Pii[idx] += lambda_o
            self.uii[idx] = self.g[idx] / self.Pii[idx]

        self.wildfire_itr(new_meas)

        self.wild_bool = False

    def update_occ(self, occ):
        self.updating_bool = True
        origin_new = self.round2grid(np.array(occ.origin), self.gridsize)
        delta = np.array((self.origin - origin_new) / self.gridsize).astype(np.uint32)
        if self.xlen != occ.xlen or self.ylen != occ.ylen or self.zlen != occ.zlen or any(delta != 0):
            self.xlen = occ.xlen
            self.ylen = occ.ylen
            self.zlen = occ.zlen

            self.origin = origin_new

            self.width = occ.w
            self.length = occ.l
            self.height = occ.h

            self.mean = np.zeros((self.xlen, self.ylen, self.zlen))
            self.mean_norm = np.zeros((self.xlen, self.ylen, self.zlen))
            self.mean_projected = np.zeros((self.xlen, self.ylen))
            self.var = np.zeros((self.xlen, self.ylen, self.zlen))
            self.var_norm = np.zeros((self.xlen, self.ylen, self.zlen))
            self.var_projected = np.zeros((self.xlen, self.ylen))

            self.map2graph = np.array(np.ones((self.xlen, self.ylen, self.zlen)) * -1).astype(int)
            for i in range(0, len(self.graph2map)):
                self.graph2map[i][0] += delta[0]
                self.graph2map[i][1] += delta[1]
                self.graph2map[i][2] += delta[2]
                if all(np.array(self.graph2map[i]) < [self.xlen, self.ylen, self.zlen]):
                    self.map2graph[self.graph2map[i][0], self.graph2map[i][1], self.graph2map[i][2]] = i
                else:
                    for child in self.Neighbours[i]:
                        self.deledge(i, child)

            for z_id in range(0, len(self.z)):
                self.z_pos[z_id][0] += delta[0]
                self.z_pos[z_id][1] += delta[1]
                self.z_pos[z_id][2] += delta[2]

        self.occ = occ.data
        self.isexpanded = [False] * self.N

        self.updating_bool = False

    def wildfire_itr(self, num_bombs):
        msgs = 0
        Q = []
        wild_start = rospy.get_time()

        for firebomb in range(1, num_bombs + 1):
            bomb_pos = self.z_graph_id[-firebomb]
            Q.append(bomb_pos)
            while Q:
                sender = Q[0]
                Q.pop(0)

                self.P[sender][0] = self.Pii[sender] + np.sum(self.P[sender][1:])
                self.u[sender][0] = 1 / self.P[sender][0] * (self.Pii[sender] * self.uii[sender] + np.sum(
                    np.array(self.P[sender][1:]) * np.array(self.u[sender][1:])))

                self.resid[sender] = 0

                if self.Neighbours[sender]:
                    j = 1
                    while j < len(self.Neighbours[sender]) + 1:
                        j_id = self.Neighbours[sender][j - 1]
                        j2s_id = self.Neighbours[j_id].index(sender) + 1

                        P_k = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k = np.float64(self.u[j_id][j2s_id])

                        self.P[j_id][j2s_id] = -self.H_p[sender][j] ** 2 / (self.P[sender][0] - self.P[sender][j])
                        self.u[j_id][j2s_id] = (self.P[sender][0] * self.u[sender][0] - self.P[sender][j] *
                                                self.u[sender][j]) / self.H_p[sender][j]

                        P_k2 = np.float64(abs(self.P[j_id][j2s_id]))
                        u_k2 = np.float64(self.u[j_id][j2s_id])

                        self.msgs_wild += 1

                        if self.dm == 0:
                            self.resid[j_id] = max(abs(u_k2 - u_k), self.resid[j_id])
                        elif self.dm == 1:
                            self.resid[j_id] = 0.25 * math.log(0.25 * (P_k2 / P_k + P_k / P_k2 + 2)) + 0.25 * (
                                (u_k - u_k2) ** 2 / (1 / P_k + 1 / P_k2))
                        elif self.dm == 2:
                            self.resid[j_id] = abs(u_k2 - u_k) / P_k

                        if self.resid[j_id] > self.eps:
                            if self.isexpanded[j_id] is False:
                                self.expandnode(j_id)
                            if j_id not in Q:
                                Q.append(j_id)
                        else:
                            converged = True
                        j += 1

                if rospy.get_time() - wild_start > self.z_freq:
                    Q = []

        done = True

    def get_means(self):
        for i in range(0, self.N):
            [x, y, z] = self.graph2map[i]
            self.mean[x, y, z] = -self.u[i][0]

        for x in range(0, self.xlen):
            for y in range(0, self.ylen):
                self.mean_projected[x, y] = np.mean(self.mean[x, y, :])

        if self.mean.max() > 0:
            self.mean_norm = self.mean / self.mean.max()

    def get_uncertainty(self):
        for i in range(0, self.N):
            if self.P[i][0] != 0:
                [x, y, z] = self.graph2map[i]
                self.var[x, y, z] = 1 / self.P[i][0]

        for x in range(0, self.xlen):
            for y in range(0, self.ylen):
                self.var_projected[x, y] = np.mean(self.var[x, y, :])

        if self.var.max() > 0:
            self.var_norm = self.var / self.var.max()

    def show_means(self, handle=1):

        fig = plt.figure(handle)
        plt.clf()
        cax = plt.matshow(self.mean, fig, vmin=0, extent=[0, self.xlen, self.ylen, 0], cmap=self.cmap)
        fig.colorbar(cax)
        plt.show(block=False)
        plt.pause(0.001)

    def show_uncertainty(self, handle=2):

        fig = plt.figure(handle)
        plt.clf()
        cax = plt.matshow(self.uncertainty, fig, vmin=0, vmax=5, extent=[0, self.width, self.height, 0], cmap=self.cmap)
        fig.colorbar(cax)
        plt.show(block=False)
        plt.pause(0.001)

    def publish_meanimage(self):
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = self.ylen
        msg.width = self.xlen
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        if self.mean_projected.max() != 0:
            im_mat = np.array(
                self.cmap(np.transpose(self.mean_projected) / self.mean_projected.max()) * [255, 255, 255, 1], dtype=np.uint8)
        else:
            im_mat = np.zeros((msg.height, msg.width, 4), dtype=np.uint8)
            zero_colour = [255, 255, 255, 1] * np.array(self.cmap(0))
            im_mat[:, :] = zero_colour
        msg.step = 3 * msg.width
        msg.data = np.flipud(im_mat[:, :, 0:3]).tobytes()
        self.meanImgPub.publish(msg)

    def publish_meanmatrix(self):
        self.meanMatrix.layout.dim[0].label = "x"
        self.meanMatrix.layout.dim[1].label = "y"
        self.meanMatrix.layout.dim[2].label = "z"
        self.meanMatrix.layout.dim[0].size = self.xlen
        self.meanMatrix.layout.dim[1].size = self.ylen
        self.meanMatrix.layout.dim[2].size = self.zlen
        self.meanMatrix.layout.dim[0].stride = self.xlen * self.ylen * self.zlen
        self.meanMatrix.layout.dim[1].stride = self.ylen * self.zlen
        self.meanMatrix.layout.dim[2].stride = self.zlen
        self.meanMatrix.layout.data_offset = 0
        self.meanMatrix.data = self.mean.flatten()
        self.meanMatPub.publish(self.meanMatrix)

    def publish_meanmarker(self):
        self.marker.points = []
        self.marker.colors = []
        for i in range(0, self.N):
            [x, y, z] = self.graph2map[i]
            alpha = self.mean_norm[x, y, z]
            if alpha >= 0.2:
                point_msg = Point()
                point_msg.x = x * self.gridsize + self.origin[0]
                point_msg.y = y * self.gridsize + self.origin[1]
                point_msg.z = z * self.gridsize + self.origin[2]
                point_colour = ColorRGBA()
                rgb = self.cmap(alpha)
                point_colour.r = rgb[0]
                point_colour.g = rgb[1]
                point_colour.b = rgb[2]
                point_colour.a = alpha
                self.marker.points.append(point_msg)
                self.marker.colors.append(point_colour)

        self.marker.header.stamp = rospy.Time.now()
        self.meanMarkerPub.publish(self.marker)

    def publish_varimage(self):
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = self.ylen
        msg.width = self.xlen
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        if self.var_projected.max() != 0:
            im_mat = np.array(
                self.cmap(np.transpose(self.var_projected)) * [255, 255, 255, 1],
                dtype=np.uint8)
        else:
            im_mat = np.zeros((msg.height, msg.width, 4), dtype=np.uint8)
            zero_colour = [255, 255, 255, 1] * np.array(self.cmap(0))
            im_mat[:, :] = zero_colour
        msg.step = 3 * msg.width
        msg.data = np.flipud(im_mat[:, :, 0:3]).tobytes()
        self.varImgPub.publish(msg)

    def publish_varmatrix(self):
        self.varMatrix.layout.dim[0].label = "x"
        self.varMatrix.layout.dim[1].label = "y"
        self.varMatrix.layout.dim[2].label = "z"
        self.varMatrix.layout.dim[0].size = self.xlen
        self.varMatrix.layout.dim[1].size = self.ylen
        self.varMatrix.layout.dim[2].size = self.zlen
        self.varMatrix.layout.dim[0].stride = self.xlen * self.ylen * self.zlen
        self.varMatrix.layout.dim[1].stride = self.ylen * self.zlen
        self.varMatrix.layout.dim[2].stride = self.zlen
        self.varMatrix.layout.data_offset = 0
        self.varMatrix.data = self.var.flatten()
        self.varMatPub.publish(self.varMatrix)

    def publish_state(self):
        msg = gabp_state()
        msg.header.stamp = rospy.Time.now()
        msg.num_nodes = self.N
        msg.num_measurements = len(self.z_pos)
        msg.msgs_wild = self.msgs_wild
        msg.msgs_residual = self.msgs_resid
        msg.msgs_total = self.msgs_resid + self.msgs_wild
        msg.max_mean = self.mean.max()
        msg.max_var = self.var.max()
        msg.max_measurement = max(self.z)
        self.statePub.publish(msg)

    def timer_callback(self, timer):
        try:
            if self.updating_bool is False:
                self.get_means()
            if self.updating_bool is False:
                self.get_uncertainty()
            #if self.mean.max() > 0:
            self.publish_meanimage()
            self.publish_meanmatrix()
            self.publish_varimage()
            self.publish_varmatrix()
            self.publish_meanmarker()
            self.publish_state()
        except:
            rospy.logwarn("!!! Error publishing ROS data, will retry !!!")

    def on_shutdown(self):
        self.stop_thread = True
