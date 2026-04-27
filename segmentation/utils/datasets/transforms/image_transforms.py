import cv2
import numpy as np
from PIL import Image
import random

from base.datasets.transforms import BaseTransforms


class ImageTransforms(BaseTransforms):
    def __init__(self, data, label=None):
        """
        :param data: requires numpy.ndarray. original image
        :param label: requires numpy.ndarray. original mask(could be None, reserved for unsupervised learning)
        """
        super().__init__(data, label)

    @staticmethod
    def __rotate(image, degree):
        """
        rotate image with degree
        :param image: requires numpy.ndarray. original image
        :param degree: requires float(0-360). degree to rotate
        :return: returns numpy.ndarray. rotated image
        """
        image = Image.fromarray(image)
        image = image.rotate(degree)
        image = np.array(image)
        return image

    @staticmethod
    def __gamma_transformation(image, gamma):
        """
        gamma transformation
        :param image: requires numpy.ndarray. original image
        :param gamma: requires float(>0). gamma
        :return: returns numpy.ndarray. transformed image
        """
        image -= image.min()
        image = image / (image.max() - image.min())
        image = np.power(image, gamma) * 255
        image = np.uint8(image)
        return image

    @staticmethod
    def __brightness_transformation(image, coefficient):
        image = np.float64(image)
        image += coefficient * np.mean(image)
        mask = image < 0
        image[mask] = 0
        mask = image > 255
        image[mask] = 255
        image = np.uint8(image)
        return image

    def resize(self, size):
        """
        resize the image and mask
        :param size: requires tuple or list(len==2). target size
        """
        self.data = cv2.resize(self.data, size)
        if self.label is not None:
            self.label = cv2.resize(self.label, size)

    def random_flip_top_bottom(self, probability=0.5):
        """
        flip from top to bottom randomly
        :param probability: requires float. probability to flip
        """
        if random.random() < probability:
            self.data = cv2.flip(self.data, 0)
            if self.label is not None:
                self.label = cv2.flip(self.label, 0)

    def random_shift(self, probability=0.9, pixel_range_percentage=(-0.3, 0.3)):
        """
        shift randomly
        :param probability: requires float. probability to shift
        :param pixel_range_percentage: requires tuple or list(len==2). range of shift percentage
        """
        if random.random() < probability:
            x_percentage = random.uniform(pixel_range_percentage[0], pixel_range_percentage[1])
            y_percentage = random.uniform(pixel_range_percentage[0], pixel_range_percentage[1])
            rows, cols = self.data.shape

            mat = np.float32([[1, 0, x_percentage * rows], [0, 1, y_percentage * cols]])
            self.data = cv2.warpAffine(self.data, mat, (rows, cols))
            if self.label is not None:
                self.label = cv2.warpAffine(self.label, mat, (rows, cols))

    def random_rotate(self, probability=0.9, degree_range=(-45, 45)):
        """
        rotate randomly
        :param probability: requires float. probability to rotate
        :param degree_range: requires tuple or list(len==2). range of rotate degree
        """
        if random.random() < probability:
            degree = random.uniform(degree_range[0], degree_range[1])
            degree = (degree + 360) % 360
            self.data = self.__rotate(self.data, degree)
            if self.label is not None:
                self.label = self.__rotate(self.label, degree)

    def random_gamma_transform(self, probability=0.9, gamma_range=(0.5, 1.5)):
        """
        gamma transform randomly
        :param probability: requires float. probability to transform
        :param gamma_range: requires tuple or list(len==2). range of gamma
        """
        if random.random() < probability:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            self.data = self.__gamma_transformation(self.data, gamma)

    def random_brightness_transform(self, probability=0.5, coefficient_range=(-0.5, 0.5)):
        """
        gamma transform randomly
        :param probability: requires float. probability to transform
        :param coefficient_range: requires tuple or list(len==2). range of coefficient
        """
        if random.random() < probability:
            coefficient = random.uniform(coefficient_range[0], coefficient_range[1])
            self.data = self.__brightness_transformation(self.data, coefficient)

    def random_resize(self, probability=0.5, image_range=((512, 1024), (512, 1024))):
        """
        :param probability: requires float. probability to transform
        :param image_range: requires tuple of tuple. range of image size
        """
        if random.random() < probability:
            x_size = random.uniform(image_range[0][0], image_range[0][1])
            y_size = random.uniform(image_range[1][0], image_range[1][1])
            origin_size = self.data.shape
            self.data = cv2.resize(self.data, (int(x_size), int(y_size)))
            self.data = cv2.resize(self.data, origin_size)
            if self.label is not None:
                self.label = cv2.resize(self.label, (int(x_size), int(y_size)))
                self.label = cv2.resize(self.label, origin_size)
