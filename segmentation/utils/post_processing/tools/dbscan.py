import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def cluster(prediction_dir_list, image_size=(1024, 1024), eps=25, min_pts=70):
    """
    :param prediction_dir_list: requires list. list of the directories of predictions, list length larger than 1 means requiring ensemble
    :param image_size: requires tuple. image size
    :param eps: requires int. max distance
    :param min_pts: requires int. min number of points
    :return: returns 3 lists. coordinates of the masks, cluster results, image names.
    """
    image_names = os.listdir(prediction_dir_list[0])
    image_names = sorted(image_names, key=lambda x: x[:-4])

    mask_coords_list = []
    cluster_list = []
    for image_name in image_names:
        if not image_name.endswith("png") and not image_name.endswith("jpg"):
            continue
        mask_coords = []
        flag = True
        for prediction_dir in prediction_dir_list:
            image_path = prediction_dir + image_name
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, image_size)
            mask_coord = image.nonzero()
            mask_coord = [[mask_coord[0][i], mask_coord[1][i]] for i in range(len(mask_coord[0]))]
            mask_coord = np.array(mask_coord)
            mask_coords.append(mask_coord)

            if len(mask_coord) == 0:
                flag = False

        mask_coords = np.concatenate(mask_coords, axis=0) if flag else []
        mask_coords_list.append(mask_coords)

    for mask_coords in mask_coords_list:
        model = DBSCAN(eps=eps, min_samples=min_pts)
        if len(mask_coords) != 0:
            result = model.fit_predict(mask_coords) + 1
            cluster_list.append(result)
        else:
            cluster_list.append([])

    return mask_coords_list, cluster_list, image_names
