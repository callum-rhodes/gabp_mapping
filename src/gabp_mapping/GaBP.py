import numpy as np
import matplotlib.pyplot as plt
import math

import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from gabp_mapping.msg import gabp_state

class G_GaBP_2D:
    def __init__(self, occ, map_frame='/map', lambda_p=0.5, lambda_o=10.0, lambda_d=1e-4, lambda_t=1e6, epsilon=1e-4, distance='euclidean', pub_freq=1):
        self.time_init = []
        self.time_add = []
        self.time_solve = []
        self.cmap = plt.get_cmap("viridis")

        self.frame = map_frame
        self.meanImgPub = rospy.Publisher('gabp/mean/img', Image, queue_size=1)
        self.meanMarkerPub = rospy.Publisher('gabp/mean/marker', Marker, queue_size=1)
        rospy.loginfo('Publishing mean map on topic: {}'.format('/gabp/mean'))
        self.varImgPub = rospy.Publisher('/gabp/var/img', Image, queue_size=1)
        rospy.loginfo('Publishing uncertainty map on topic: {}'.format('/gabp/var'))
        self.statePub = rospy.Publisher('gabp/state', gabp_state, queue_size=1)
        rospy.loginfo('Publishing state on topic: {}'.format('/gabp/state'))
        self.rate = rospy.Duration(pub_freq)
        self.wild_bool = True
        self.updating_bool = False
        self.markerArray = MarkerArray()

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
        self.map2graph = np.array(np.ones((self.xlen, self.ylen)) * -1).astype(int)
        self.graph2map = []
        self.gridsize = occ.gridsize
        self.origin = G_GaBP_2D.round2grid(np.array(occ.origin), self.gridsize)
        self.width = occ.w
        self.length = occ.l

        self.N = int(0)
        self.H_p = []
        self.g = []

        self.Neighbours = []
        self.P = []
        self.u = []
        self.resid = []
        self.isexpanded = []
        self.msgs_wild = 0
        self.msgs_resid = 0

        self.z = []
        self.z_pos = []
        self.z_time = []
        self.z_graph_id = []

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

        self.Pii = []
        self.uii = []

        self.timer = rospy.Timer(self.rate, self.timer_callback)

        np.seterr(divide='ignore')

    @staticmethod
    def round2grid(array, nearest):
        rounded = np.round(array / nearest) * nearest
        return rounded

    def addedge(self, parent_id, child):
        if self.map2graph[child[0], child[1]] == -1:
            self.N += 1
            child_id = self.N - 1
            self.map2graph[child[0], child[1]] = child_id
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
            child_id = self.map2graph[child[0], child[1]]
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
            child_id = self.map2graph[child[0], child[1]]
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
        self.graph2map.append([x, y])
        self.map2graph[x, y] = idx
        if self.occ[x, y] == 0:
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

    def expandnode(self, id):
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

    def residual_loop(self):
        np.seterr(divide='ignore')
        while True:
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
            if all(z_pos_grid >= 0) and all(z_pos_grid[0:2] < [self.xlen, self.ylen]):
                self.z_pos.append(z_pos_grid[0:2])
                self.z_time.append(measurement.timestamp[new_z_id])
                self.z.append(measurement.data[new_z_id])
                self.z_graph_id.append(-1)
            else:
                rospy.loginfo("!!! measurement out of bounds !!!")
                rospy.loginfo("!!! skipping measurement !!!")
                new_meas -= 1

        for z_id in range(0, len(self.z)):
            if self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1]] == -1:  # cell factor already exists
                self.addnode(self.z_pos[z_id])
            idx = self.map2graph[self.z_pos[z_id][0], self.z_pos[z_id][1]]
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
            for i in range(0, len(self.graph2map)):
                self.graph2map[i][0] += delta[0]
                self.graph2map[i][1] += delta[1]
                if all(np.array(self.graph2map[i]) < [self.xlen, self.ylen]):
                    self.map2graph[self.graph2map[i][0], self.graph2map[i][1]] = i
                else:
                    for child in self.Neighbours[i]:
                        self.deledge(i, child)

            for z_id in range(0, len(self.z)):
                self.z_pos[z_id][0] += delta[0]
                self.z_pos[z_id][1] += delta[1]

        self.occ = occ.data
        self.isexpanded = [False] * self.N

        self.updating_bool = False

    def wildfire_itr(self, num_bombs):
        msgs = 0
        Q = []

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

        done = True

    def get_means(self):
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                idx = self.map2graph[x, y]
                if idx != -1:
                    self.mean[x, y] = -self.u[idx][0]
        if self.mean.max() > 0:
            self.mean_norm = self.mean / self.mean.max()

    def get_uncertainty(self):
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                idx = self.map2graph[x, y]
                if idx != -1:
                    if self.P[idx][0] != 0:
                        self.var[x, y] = min(1 / self.P[idx][0], 1)
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
        # z = int(math.floor(self.mean.shape[2] / 2))
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

    def publish_meanmarker(self):
        self.marker.points = []
        self.marker.colors = []
        for y in range(0, self.ylen):
            for x in range(0, self.xlen):
                alpha = self.mean_norm[x, y]
                if alpha >= 0.1:
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

    def publish_varimage(self):
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
            self.publish_varimage()
            self.publish_meanmarker()
            self.publish_state()

        except:
            rospy.loginfo("!! Publisher Died !!")


class G_GaBP_3D:
    def __init__(self, occ, map_frame='/map', lambda_p=0.5, lambda_o=10.0, lambda_d=1e-4, lambda_t=1e6, epsilon=1e-4, distance='euclidean', pub_freq=1):
        self.time_init = []
        self.time_add = []
        self.time_solve = []
        self.cmap = plt.get_cmap("viridis")

        self.frame = map_frame
        self.meanImgPub = rospy.Publisher('gabp/mean/img', Image, queue_size=1)
        self.meanMarkerPub = rospy.Publisher('gabp/mean/marker', Marker, queue_size=1)
        rospy.loginfo('Publishing mean map on topic: {}'.format('/gabp/mean'))
        self.varImgPub = rospy.Publisher('/gabp/var/img', Image, queue_size=1)
        rospy.loginfo('Publishing uncertainty map on topic: {}'.format('/gabp/var'))
        self.statePub = rospy.Publisher('gabp/state', gabp_state, queue_size=1)
        rospy.loginfo('Publishing state on topic: {}'.format('/gabp/state'))
        self.rate = rospy.Duration(pub_freq)
        self.wild_bool = True
        self.updating_bool = False
        self.markerArray = MarkerArray()

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
        while True:
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
                rospy.loginfo("!!! measurement out of bounds !!!")
                rospy.loginfo("!!! skipping measurement !!!")
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
        # z = int(math.floor(self.mean.shape[2] / 2))
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

    def publish_meanmarker(self):
        self.marker.points = []
        self.marker.colors = []
        for i in range(0, self.N):
            [x, y, z] = self.graph2map[i]
            alpha = self.mean_norm[x, y, z]
            if alpha >= 0.1:
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
            self.publish_varimage()
            self.publish_meanmarker()
            self.publish_state()
        except:
            rospy.loginfo("!! Publisher Died !!")

