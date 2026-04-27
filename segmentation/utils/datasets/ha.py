import cv2
import os
import torch
from torch.utils.data import Dataset
from .transforms import ImageTransforms


class HaDataset(Dataset):
    @staticmethod
    def __path_unify(path):
        """
        Unify the path with '/' at end
        :param path: data path
        :return: path that ends with '/'
        """
        if not path.endswith('/'):
            path = path + '/'
        return path

    def __init__(self,
                 data_dir,
                 image_folder_name='images/',
                 mask_folder_name='masks/',
                 imread_config=cv2.IMREAD_GRAYSCALE,
                 image_size=(1024, 1024),
                 test_mode=False):
        super().__init__()
        data_dir = self.__path_unify(data_dir)
        image_folder_name = self.__path_unify(image_folder_name)
        mask_folder_name = self.__path_unify(mask_folder_name)

        image_dir = data_dir + image_folder_name
        self.image_paths = os.listdir(image_dir)
        self.image_paths = [image_dir + image_name for image_name in self.image_paths]

        mask_dir = data_dir + mask_folder_name
        self.mask_paths = os.listdir(mask_dir)
        self.mask_paths = [mask_dir + mask_name for mask_name in self.mask_paths]

        self.imread_config = imread_config
        self.image_size = image_size
        self.test_mode = test_mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = cv2.imread(image_path, self.imread_config)
        mask = cv2.imread(mask_path, self.imread_config)

        trans = ImageTransforms(image, mask)
        trans.resize(self.image_size)

        # Data Augmentation
        if not self.test_mode:
            trans.random_flip_top_bottom(probability=0.5)
            trans.random_rotate(probability=0.9, degree_range=(-45, 45))
            trans.random_gamma_transform(probability=0.9, gamma_range=(0.5, 1.5))
            trans.random_resize(probability=0.6, image_range=((511, 1024), (511, 1024)))

        trans.min_max_normalization()
        trans.to_tensor(torch.float32)

        image = trans.data
        mask = trans.label

        mask = mask / mask.max() * 1 if mask.max() != 0 else mask

        return image, mask
