import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor, ControlPoint
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
import math


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
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    # alternative method wikipedia
    voxel = (np.array([0,0,0]) * (1 - x) * (1 - y) * (1 - z) +
    np.array([1, 0, 0]) *  x * (1 - y) * (1 - z) +
    np.array([0, 1, 0]) * (1 - x) * y * (1 - z) +
    np.array([0, 0, 1]) * (1 - x) * (1 - y) * z +
    np.array([1, 0, 1]) * x * (1 - y) * z +
    np.array([0, 1, 1]) * (1 - x) * y * z +
    np.array([1, 1, 0]) * x * y * (1 - z) +
    np.array([1, 1, 1]) * x * y * z)

    coX = math.floor(voxel[0])
    coY = math.floor(voxel[1])
    coZ = math.floor(voxel[2])

    return volume.data[coX, coY, coZ]



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

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                values = []
                for k in range(0, image_size, step):
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

                    # Break when max voxel value is found
                    if value == 205:
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
    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray, trilinear: bool=False):
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
        print(volume_maximum)

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                colors = TFColor() # initialize color
                for k in range(0, image_size, 50):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                         view_vector[0] * (k - image_center) + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                         view_vector[1] * (k - image_center) + volume_center[1]

                    # Get the voxel coordinate Y
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                         view_vector[2] * (k - image_center) + volume_center[2]
                    # Get voxel value
                    if trilinear:
                        value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                        voxel_coordinate_z)
                    else:
                        value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                    # Compositing function
                    new_color = self.tfunc.get_color(value)
                    colors.r = new_color.a * new_color.r + (1 - new_color.a) * colors.r
                    colors.g = new_color.a * new_color.g + (1 - new_color.a) * colors.g
                    colors.b = new_color.a * new_color.b + (1 - new_color.a) * colors.b
                    colors.a = new_color.a + (1 - new_color.a) * colors.a

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
                           image_size: int, image: np.ndarray, trilinear: bool=False):
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

        volume = annotation_volume

        import matplotlib.pyplot as plt
        volumes = []
        #volumes.append(annotation_volume)
        for x in energy_volumes:
            volume = energy_volumes[x]
            volumes.append(volume)
            #print(volume.get_maximum())
            data = volume.data
            histogram = np.histogram(volume.data, bins=np.arange(volume.get_maximum() + 1))[0]
            #print(histogram)

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        #self.tfunc.init(volume.get_minimum(), volume.get_maximum())

        # Define a step size to make the loop faster
        step = 2 if self.interactive_mode else 1
        step_k = 50

        #color
        colorRed = [255 / 255, 0 / 255, 0 / 255]

        colorLila = [127 / 255, 0 / 255, 255 / 255]
        colorPurple = [255 / 255, 0 / 255, 127 / 255]
        colorBlue = [0 / 255, 0 / 255, 255 / 255]
        colorGreen = [0 / 255, 255/ 255, 0 / 255]
        colorYellow = [255 / 255, 255 / 255, 0 / 255]
        colorBrown = [255 / 255, 127 / 255, 0 / 255]
        colorLightgreen = [127/ 255, 255 / 255, 0 / 255]
        colorLightBlue = [0  / 255, 127 / 255, 255 / 55]
        colortorqoise = [0 / 255, 255 / 255, 255/ 255]
        colors = [colorLila, colorLightgreen, colorPurple, colorBlue, colorGreen, colorYellow, colortorqoise, colorLightgreen, colorRed]


        for i in range(0, image_size, step):
            for j in range(0, image_size, step):
                counter = 0
                compValue = 0
                for x in volumes:
                    volume = x
                    compValueVol = 0
                    for k in range(0, image_size, step_k):
                        # Get the voxel coordinate X
                        voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                             view_vector[0] * (k - image_center) + volume_center[0]

                        # Get the voxel coordinate Y
                        voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                             view_vector[1] * (k - image_center) + volume_center[1]

                        # Get the voxel coordinate Y
                        voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                             view_vector[2] * (k - image_center) + volume_center[2]
                        # Get voxel value
                        if trilinear:
                            value = trilinear_interpolation(volume, voxel_coordinate_x, voxel_coordinate_y,
                                                            voxel_coordinate_z)
                        else:
                            value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                        # Deal with big values in annotation
                        #if counter == 0:
                        #    if value > 0:
                        #        value = 5

                        # Learning function
                        compValueVol = 0.15 * value + 0.85 * compValue

                    # Take the highest value
                    if compValueVol > compValue:
                        compValue = compValueVol
                        setColor = colors[counter]
                    # Color pixel if there is a value
                    if compValue > 0.1 :
                        red = setColor[0]
                        green = setColor[1]
                        blue = setColor[2]
                        alpha = compValue
                        print(alpha)
                    else:
                        red = 0
                        green = 0
                        blue = 0
                        alpha = 0
                    counter += 1
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
