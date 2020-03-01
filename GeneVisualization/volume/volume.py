import numpy as np
import math


class Volume:
    """
    Volume data class.

    Attributes:
        data: Numpy array with the voxel data. Its shape will be (dim_x, dim_y, dim_z).
        dim_x: Size of dimension X.
        dim_y: Size of dimension Y.
        dim_z: Size of dimension Z.
    """

    def __init__(self, array, compute_histogram=True):
        """
        Inits the volume data.
        :param array: Numpy array with shape (dim_x, dim_y, dim_z).
        """

        self.data = array
        self.histogram = np.array([])
        self.dim_x = array.shape[0]
        self.dim_y = array.shape[1]
        self.dim_z = array.shape[2]

        if compute_histogram:
            self.compute_histogram()

    def get_voxel(self, x, y, z):
        """Retrieves the voxel for the """
        return self.data[x, y, z]

    def get_minimum(self):
        return self.data.min()

    def get_maximum(self):
        return self.data.max()

    def compute_histogram(self):
        self.histogram = np.histogram(self.data, bins=np.arange(self.get_maximum() + 1))[0]


class VoxelGradient:
    def __init__(self, gx=0, gy=0, gz=0):
        self.x = gx
        self.y = gy
        self.z = gz
        self.magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)


ZERO_GRADIENT = VoxelGradient()


class GradientVolume:
    def __init__(self, volume):
        self.volume = volume
        self.data = []
        self.compute()
        self.max_magnitude = -1.0

    def get_gradient(self, x, y, z):
        return self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)]

    def set_gradient(self, x, y, z, value):
        self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)] = value

    def get_voxel(self, i):
        return self.data[i]

    def compute(self):
        """
        Computes the gradient for the current volume
        """
        # this just initializes all gradients to the vector (0,0,0)
        self.data = [ZERO_GRADIENT.magnitude] * (self.volume.dim_x * self.volume.dim_y * self.volume.dim_z)
        # gm = []
        # vv = []
        for i in range(1, self.volume.dim_x-1):
            for j in range(1, self.volume.dim_y-1):
                for k in range(1, self.volume.dim_z-1):
                    s1_x = self.volume.data[i - 1, j, k]
                    s2_x = self.volume.data[i + 1, j, k]
                    s_x = (s2_x - s1_x) / 2
                    s1_y = self.volume.data[i, j - 1, k]
                    s2_y = self.volume.data[i, j + 1, k]
                    s_y = (s2_y - s1_y) / 2
                    s1_z = self.volume.data[i, j, k - 1]
                    s2_z = self.volume.data[i, j, k + 1]
                    s_z = (s2_z - s1_z) / 2
                    s = np.array([s_x, s_y, s_z])
                    if np.linalg.norm(s)!=0:
                        normalized = s / np.linalg.norm(s)
                    else:
                        normalized = np.array([0, 0, 0])
                    if np.isnan(normalized).any():
                        normalized = np.array([0, 0, 0])
                    self.data[i + self.volume.dim_x * (j + self.volume.dim_y * k)] = \
                        VoxelGradient(normalized[0], normalized[1], normalized[2]).magnitude
                    # gm.append(self.data[i + self.volume.dim_x * (j + self.volume.dim_y * k)].magnitude)
                    # vv.append(self.volume.data[i, j, k])
                    # if self.data[i + self.volume.dim_x * (j + self.volume.dim_y * k)].magnitude!=0:
                    #     print("scalar volume value: ", self.volume.data[i, j, k])
                    #     print("gradient magnitude: ", self.data[i + self.volume.dim_x * (j + self.volume.dim_y * k)].magnitude)
        print(len(self.data))
        # print("range volume value: ", min(vv), "-", max(vv))
        # print("range gradient magnitude: ", min(gm), "-", max(gm))
        # ind = [i for i, j in enumerate(gm) if j == max(gm)]
        # maxg_vv = [vv[i] for i in ind]
        # print("volume values where gradient magnitude is max: ", maxg_vv)
        # print(self.data)

    def get_max_gradient_magnitude(self):
        if self.max_magnitude < 0:
            gradient = max(self.data, key=lambda x: x.magnitude)
            self.max_magnitude = gradient.magnitude

        return self.max_magnitude
