import numpy as np
from PIL import Image
import math


class OccGenerator:
    def __init__(self):
        self.data = []
        self.xlen = []
        self.ylen = []
        self.zlen = []
        self.gridsize = []
        self.w = []
        self.l = []
        self.h = []
        self.origin = [0, 0]
        self.max = [0, 0]

    def mat2occ(self, data, gridsize, xlen=None, ylen=None):
        if xlen is not None:
            self.data = np.array(data).reshape(xlen, ylen)
            self.data = np.round(self.data / np.amax(self.data))
            self.data = self.data.astype(np.int)
            self.xlen = xlen
            self.ylen = ylen
        else:
            self.data = np.round(data / np.amax(data))
            self.data = self.data.astype(np.int)
            size = data.shape
            self.ylen = size[0]
            self.xlen = size[1]

        self.w = self.xlen * gridsize
        self.h = self.ylen * gridsize

        self.gridsize = gridsize

    def img2occ(self, img, gridsize, size=None):
        im = Image.open(img)
        if size is not None:
            im = im.resize(size, resample=1)
        im = np.array(im)
        im_bin = im[:, :, 0]
        im_bin = 1-(np.round(im_bin / np.amax(im_bin)))
        self.data = im_bin.astype(np.int)
        self.ylen = size[1]
        self.xlen = size[0]
        self.gridsize = gridsize

        self.w = self.xlen * gridsize
        self.h = self.ylen * gridsize

    def ros2occ(self, occ, des_res=None):
        self.xlen = occ.info.width
        self.ylen = occ.info.height
        if des_res is not None:
            scale = occ.info.resolution / des_res
            size = [int(self.xlen * scale), int(self.ylen * scale)]
            mat = np.array(occ.data).astype(dtype=np.uint8).reshape(self.ylen, self.xlen)
            im = Image.fromarray(mat)
            im = im.resize(size, resample=0)
            self.data = np.array(im).astype(dtype=np.uint32) * 100 / 255
            self.data = np.transpose(self.data)
            self.data[self.data > 0] = 1
            self.xlen = size[0]
            self.ylen = size[1]
            self.gridsize = des_res

        else:
            self.data = np.array(occ.data).reshape(self.ylen, self.xlen)
            self.data = np.transpose(self.data)
            self.gridsize = np.array(occ.info.resolution)

        self.w = self.xlen * self.gridsize
        self.h = self.ylen * self.gridsize
        self.origin = np.array([occ.info.origin.position.x, occ.info.origin.position.y, 0])

    def oct2occ(self, oct, des_res=None):
        size = oct.getMetricSize()
        self.w = size[0]
        self.l = size[1]
        self.h = size[2]

        self.origin = oct.getMetricMin()
        self.max = self.origin + size

        if des_res is not None:
            self.gridsize = des_res
        else:
            self.gridsize = 1

        self.xlen = int(math.floor(self.w / self.gridsize))
        self.ylen = int(math.floor(self.l / self.gridsize))
        self.zlen = int(math.floor(self.h / self.gridsize))

        self.data = np.empty((self.xlen, self.ylen, self.zlen), dtype=np.int8)

        for z in range(0, self.zlen):
            for y in range(0, self.ylen):
                for x in range(0, self.xlen):
                    point = np.array([self.origin[0] + (x+0.5) * self.gridsize, self.origin[1] + (y+0.5) * self.gridsize, self.origin[2] + (z+0.5) * self.gridsize])
                    node = oct.search(point)
                    try:
                        bool = oct.isNodeOccupied(node)
                    except:
                        bool = -1

                    self.data[x, y, z] = bool

    def blank2occ(self, size, res):
        self.data = np.zeros(size)
        self.xlen = size[0]
        self.ylen = size[1]
        self.origin = [0, 0, 0]
        if len(size) == 3:
            self.zlen = size[2]
        self.gridsize = res
