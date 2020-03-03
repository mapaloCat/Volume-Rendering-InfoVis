import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor, ControlPoint
from volume.volume import GradientVolume, Volume, VoxelGradient
from collections.abc import ValuesView
import math
import matplotlib.pyplot as plt
from wx import Colour

# TODO: Implement trilinear interpolation
def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))

    return volume.data[x, y, z]


def trilinear_interpolation(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates using trilinear interpolation.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """

    # Keep object between boundaries
    # if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
    #     return 0

    # alternative method wikipedia
    # voxel = (np.array([0, 0, 0]) * (1 - x) * (1 - y) * (1 - z) +
    # np.array([1, 0, 0]) * x * (1 - y) * (1 - z) +
    # np.array([0, 1, 0]) * (1 - x) * y * (1 - z) +
    # np.array([0, 0, 1]) * (1 - x) * (1 - y) * z +
    # np.array([1, 0, 1]) * x * (1 - y) * z +
    # np.array([0, 1, 1]) * (1 - x) * y * z +
    # np.array([1, 1, 0]) * x * y * (1 - z) +
    # np.array([1, 1, 1]) * x * y * z)
    #
    # coX = math.floor(voxel[0])
    # coY = math.floor(voxel[1])
    # coZ = math.floor(voxel[2])
    #
    # return volume.data[coX, coY, coZ]

    ax = x
    ay = y
    az = z

    if ax < 0 or ay < 0 or az < 0 or ax >= volume.dim_x-1 or ay >= volume.dim_y-1 or az >= volume.dim_z-1:
        return 0

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))

    c000 = volume.get_voxel(x, y, z)
    c100 = volume.get_voxel(x + 1, y, z)
    c010 = volume.get_voxel(x, y + 1, z)
    c110 = volume.get_voxel(x + 1, y + 1, z)
    c001 = volume.get_voxel(x, y, z + 1)
    c101 = volume.get_voxel(x + 1, y, z + 1)
    c011 = volume.get_voxel(x, y + 1, z + 1)
    c111 = volume.get_voxel(x + 1, y + 1, z + 1)

    xd = ax - x
    yd = ay - y
    zd = az - z
    # four linear interpolation
    c00 = xd * c100 + (1 - xd) * c000
    c01 = xd * c101 + (1 - xd) * c001
    c10 = xd * c110 + (1 - xd) * c010
    c11 = xd * c111 + (1 - xd) * c011
    # two bi-linear interpolation
    c0 = yd * c10 + (1 - yd) * c00
    c1 = yd * c11 + (1 - yd) * c01
    # one tri-linear interpolation
    c = zd * c1 + (1 - zd) * c0

    return math.floor(c)


def getVoxelGradient(volume: Volume, gradients: GradientVolume, x: float, y: float, z: float):
    """
    Retrieves the gradient of a voxel for the given coordinates.
    :param volume: Volume from which the voxel gradient will be retrieved.
    :param gradients: GradientVolume from which the voxel gradient will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel gradient
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x-1 or y >= volume.dim_y-1 or z >= volume.dim_z-1:
        return VoxelGradient()

    x = int(math.floor(x))
    y = int(math.floor(y))
    z = int(math.floor(z))

    return gradients.get_gradient(x, y, z)


class RaycastRendererImplementation(RaycastRenderer):
    """
    Class to be implemented.
    """

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, trilinear: bool=False):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                # Get the voxel coordinate X
                voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                     volume_center[0]

                # Get the voxel coordinate Y
                voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                     volume_center[1]

                # Get the voxel coordinate Z
                voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                     volume_center[2]
                # Get voxel value
                if trilinear:
                    value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                    voxel_coordinate_z)
                else:
                    value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Implement MIP function
    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, trilinear: bool=False):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1
        step_k = 10

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                values = []
                for k in range(0, image_size, step_k):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         view_vector[0] * (k - image_center) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         view_vector[1] * (k - image_center) + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         view_vector[2] * (k - image_center) + volume_center[2]
                    # Get voxel value
                    if trilinear:
                        value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                        voxel_coordinate_z)
                    else:
                        value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    values.append(value)

                    # Break when max voxel value is reached
                    if value == volume_maximum:
                        break

                # Normalize value to be between 0 and 1
                red = max(values) / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Implement Compositing function. TFColor is already imported. self.tfunc is the current transfer function.
    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, trilinear: bool=True, shading: bool=True):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1
        step_k = 2

        if shading:
            # ambient reflection constant, the ratio of reflection of the ambient term present in all points in the scene rendered
            # we assume the light source is white
            shading_ambient_coeff = TFColor(0.1, 0.1, 0.1, 1.0)
            # diffuse reflection constant, the ratio of reflection of the diffuse term of incoming light
            shading_diff_coeff = 0.7 # 0.5
            # specular reflection constant, the ratio of reflection of the specular term of incoming light
            shading_spec_coeff = 0.2 # 0.4
            # shininess constant for this material, which is larger for surfaces that are smoother and more mirror-like
            shading_alpha = 10
            # compute volume gradients
            gradients = GradientVolume(volume)

        borders = math.sqrt(math.pow(volume.dim_x * view_vector[0], 2)
                            + math.pow(volume.dim_y * view_vector[1], 2)
                            + math.pow(volume.dim_z * view_vector[2], 2))

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                colors = TFColor(0, 0, 0, 1) # initialize color
                voxel_color = TFColor()
                # for k in range(0, image_size, step_k):
                for k in range(int(borders/2), -int(borders/2), -step_k):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         view_vector[0] * k + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         view_vector[1] * k + volume_center[1]

                    # Get the voxel coordinate Y
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         view_vector[2] * k + volume_center[2]
                    # Get voxel value
                    if trilinear:
                        value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                        voxel_coordinate_z)
                    else:
                        value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                    # Compositing function
                    # new_color = self.tfunc.get_color(value)
                    # colors.r = new_color.a * new_color.r + (1 - new_color.a) * colors.r
                    # colors.g = new_color.a * new_color.g + (1 - new_color.a) * colors.g
                    # colors.b = new_color.a * new_color.b + (1 - new_color.a) * colors.b
                    # colors.a = new_color.a + (1 - new_color.a) * colors.a
                    tf_color = self.tfunc.get_color(value)
                    voxel_color.r = tf_color.r
                    voxel_color.g = tf_color.g
                    voxel_color.b = tf_color.b
                    voxel_color.a = tf_color.a
                    if shading:
                        vg = getVoxelGradient(volume, gradients, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                        # dotproductln = (view_vector[0] * vg.x + view_vector[1] * vg.y + view_vector[2] * vg.z)
                        dotproductln = np.dot(view_vector, np.array([vg.x, vg.y, vg.z]))
                        if dotproductln > 0:
                            ln = dotproductln/vg.magnitude
                            vr = math.pow(ln, shading_alpha)
                            voxel_color.r = shading_ambient_coeff.r + voxel_color.r * shading_diff_coeff * ln + shading_spec_coeff * vr
                            voxel_color.g = shading_ambient_coeff.g + voxel_color.g * shading_diff_coeff * ln + shading_spec_coeff * vr
                            voxel_color.b = shading_ambient_coeff.b + voxel_color.b * shading_diff_coeff * ln + shading_spec_coeff * vr
                    colors.r = (1 - voxel_color.a) * colors.r + voxel_color.a * voxel_color.r
                    colors.g = (1 - voxel_color.a) * colors.g + voxel_color.a * voxel_color.g
                    colors.b = (1 - voxel_color.a) * colors.b + voxel_color.a * voxel_color.b
                    colors.a = (1 - voxel_color.a) * colors.a + voxel_color.a

                # Set colors
                red = colors.r
                green = colors.g
                blue = colors.b
                alpha = colors.a

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # TODO: Implement function to render multiple energy volumes and annotation volume as a silhouette.
    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray, trilinear: bool=True, shading: bool=True):
        # TODO: Implement your code considering these volumes (annotation_volume, and energy_volumes)
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        print("annotation volume info: ")
        print(annotation_volume.data.shape)

        # Center of the annotation volume (3-dimensional)
        volume_center = [annotation_volume.dim_x / 2, annotation_volume.dim_y / 2, annotation_volume.dim_z / 2]

        if shading:
            # ambient reflection constant, the ratio of reflection of the ambient term present in all points in the scene rendered
            # we assume the light source is white
            shading_ambient_coeff = TFColor(0.1, 0.1, 0.1, 1.0)
            # diffuse reflection constant, the ratio of reflection of the diffuse term of incoming light
            shading_diff_coeff = 0.7  # 0.5
            # specular reflection constant, the ratio of reflection of the specular term of incoming light
            shading_spec_coeff = 0.2  # 0.4
            # shininess constant for this material, which is larger for surfaces that are smoother and more mirror-like
            shading_alpha = 10

        max_energy_volume = []
        gradients = []
        for key, volume in energy_volumes.items():
            # print(key, volume)
            max_energy_volume.append(volume.get_maximum())
            # print(np.histogram(volume.data, bins=np.arange(volume.get_maximum() + 1))[0])
            # print(np.arange(volume.get_maximum() + 1))
            # plt.bar(np.arange(volume.get_maximum()),
            #         np.histogram(volume.data, bins=np.arange(volume.get_maximum() + 1))[0], width=0.1)
            # plt.ylim((0, 1000))
            # plt.xlim((1, volume.get_maximum()))
            # plt.show()
            print("energy volume info: ")
            print(volume.data.shape)
            print("energy volume range: ", volume.get_minimum(), "-", volume.get_maximum())
            # Initialize GradientVolume
            # print("gradient computation info: ")
            gradients.append(GradientVolume(volume))

        # palette = [Colour(255, 0, 0),
        #            Colour(0, 255, 0),
        #            Colour(0, 0, 255),
        #            Colour(255, 255, 0),
        #            Colour(255, 0, 255),
        #            Colour(0, 255, 255),
        #            Colour(128, 128, 0),
        #            Colour(240, 128, 128),
        #            Colour(152, 251, 152),
        #            Colour(139, 69, 19)]

        # Initialize TF
        self.tfunc.init(0, max(max_energy_volume))

        # set the control points
        self.tfunc.add_control_point(0, 0., .0, .0, .0)
        self.tfunc.add_control_point(2, 0., .0, .0, .0)
        self.tfunc.add_control_point(4, 1., .666, .0, 1.)
        self.tfunc.add_control_point(13, 0., 0., .0, 0.5)
        self.tfunc.add_control_point(17, 0., 0., .0, .0)
        self.tfunc.add_control_point(21, 1., .0, .0, 1.)
        self.tfunc.add_control_point(39, 0., .0, .0, 0.5)
        self.tfunc.add_control_point(87, 0., .0, .0, .0)

        # self.tfunc.update_control_point_color(2, palette[9])

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1
        step_k = 2

        borders = math.sqrt(math.pow(annotation_volume.dim_x * view_vector[0], 2)
                            + math.pow(annotation_volume.dim_y * view_vector[1], 2)
                            + math.pow(annotation_volume.dim_z * view_vector[2], 2))

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                colors = TFColor(0, 0, 0, 1)  # initialize color
                voxel_color = TFColor()
                for k in range(int(borders/2), -int(borders/2), -step_k):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         view_vector[0] * k + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         view_vector[1] * k + volume_center[1]

                    # Get the voxel coordinate Y
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         view_vector[2] * k + volume_center[2]
                    energy_values = []
                    for x in energy_volumes:
                        volume = energy_volumes[x]
                        # Get voxel value
                        if trilinear:
                            value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                            voxel_coordinate_z)
                        else:
                            value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                        energy_values.append(value)

                    # Compositing function
                    tf_color = self.tfunc.get_color(max(energy_values))
                    voxel_color.r = tf_color.r
                    voxel_color.g = tf_color.g
                    voxel_color.b = tf_color.b
                    voxel_color.a = tf_color.a
                    if shading:
                        vg = getVoxelGradient(volume, gradients[np.argmax(energy_values)], voxel_coordinate_x, voxel_coordinate_y,
                                              voxel_coordinate_z)
                        # dotproductln = (view_vector[0] * vg.x + view_vector[1] * vg.y + view_vector[2] * vg.z)
                        dotproductln = np.dot(view_vector, np.array([vg.x, vg.y, vg.z]))
                        if dotproductln > 0:
                            ln = dotproductln / vg.magnitude
                            vr = math.pow(ln, shading_alpha)
                            voxel_color.r = shading_ambient_coeff.r + voxel_color.r * shading_diff_coeff * ln + shading_spec_coeff * vr
                            voxel_color.g = shading_ambient_coeff.g + voxel_color.g * shading_diff_coeff * ln + shading_spec_coeff * vr
                            voxel_color.b = shading_ambient_coeff.b + voxel_color.b * shading_diff_coeff * ln + shading_spec_coeff * vr
                    colors.r = (1 - voxel_color.a) * colors.r + voxel_color.a * voxel_color.r
                    colors.g = (1 - voxel_color.a) * colors.g + voxel_color.a * voxel_color.g
                    colors.b = (1 - voxel_color.a) * colors.b + voxel_color.a * voxel_color.b
                    colors.a = (1 - voxel_color.a) * colors.a + voxel_color.a

                # Set colors
                red = colors.r
                green = colors.g
                blue = colors.b
                alpha = colors.a

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha


class GradientVolumeImpl(GradientVolume):
    # TODO: Implement gradient compute function. See parent class to check available attributes.
    def compute(self):
        pass
