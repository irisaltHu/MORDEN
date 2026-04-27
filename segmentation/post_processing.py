import os
import cv2
import numpy as np
from PIL import Image
from utils.post_processing.tools import cluster
import sys
from pathlib import Path


def save_cluster(mask_coord_list, cluster_list, image_names, save_dir, image_size=(1024, 1024)):
    """
    :param mask_coord_list: requires list. list of coordinates of the masks
    :param cluster_list: requires list. list of cluster results
    :param image_names: requires list. list of image names
    :param save_dir: requires str. directory to save the visualization result
    :param image_size: requires tuple. image size
    :return: None
    """
    path = Path(save_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=True)

    for i in range(len(cluster_list)):
        mask_coord = mask_coord_list[i]
        cluster_result = cluster_list[i]

        canvas = np.zeros((image_size[0], image_size[1], 3), dtype=float)
        # canvas = np.zeros(image_size, dtype=np.uint8)
        cluster_num = np.max(cluster_result) if len(cluster_result) > 0 else 0
        # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(set(cluster_result)))]
        for j in range(1, cluster_num + 1):
            mask = cluster_result == j
            coord = mask_coord[mask]
            max_x, max_y = np.max(coord, axis=0)
            min_x, min_y = np.min(coord, axis=0)
            for k in range(len(coord)):
                # canvas[coord[k][0], coord[k][1]] = j
                canvas[coord[k][0]][coord[k][1]] = (255, 255, 255)
                # canvas[coord[k][0]][coord[k][1]] = colors[j][:3]
            cv2.rectangle(canvas, (min_y, min_x), (max_y, max_x), (0, 0, 255), 2)
            cv2.putText(canvas, str(j), (min_y, min_x), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4)

        print("saving " + save_dir + image_names[i])
        cv2.imwrite(save_dir + image_names[i], np.uint8(canvas))
        # Image.fromarray(canvas).convert('L').save(save_dir + image_names[i])
        # plt.imsave(save_dir + image_names[i], canvas)


def cluster_images():
    image_size = (1024, 1024)
    inference_dir = ["output/MHAS/test/seg/"]
    mask_coord_list, cluster_list, image_names = cluster(inference_dir, image_size=image_size, eps=25, min_pts=70)
    visualization_save_dir = "output/MHAS/test/cluster/"
    save_cluster(mask_coord_list, cluster_list, image_names, visualization_save_dir, image_size=image_size)


if __name__ == "__main__":
    cluster_images()
