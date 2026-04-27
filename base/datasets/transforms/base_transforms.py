import numpy as np
import torch
from torchvision import transforms


class BaseTransforms:
    def __init__(self, data, label=None):
        """
        :param data: requires numpy.ndarray.
        :param label: requires numpy.ndarray.
        """
        self.data = data
        self.label = label

    def to_tensor(self, dtype=torch.float32):
        """
        from numpy/PIL.Image to tensor
        :param dtype: target data type
        """
        self.data = transforms.ToTensor()(self.data).to(dtype)
        if self.label is not None:
            self.label = transforms.ToTensor()(self.label).to(dtype)

    def min_max_normalization(self, label_process=False):
        """
        min-max normalization
        """
        self.data -= self.data.min()
        self.data = self.data / (self.data.max() - self.data.min())
        if label_process:
            self.label -= self.label.min()
            self.label = self.label / (self.label.max() - self.label.min())

    def mean_normalization(self):
        """
        mean normalization
        """
        self.data -= self.data.mean()
        self.data = self.data / (self.data.max() - self.data.min())

    def standardization(self, axis):
        """
        standardization
        """
        self.data -= self.data.mean()
        self.data /= np.sqrt(np.var(self.data, axis=axis))

