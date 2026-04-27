import cv2
import numpy as np


def edge_detection(mask, image=None, image_size=(1024, 1024), edge_color=(255, 0, 0)):
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    edges = cv2.Canny(mask, 10, 70)

    if image is not None:
        image = cv2.resize(image, image_size)
        mask = edges == 255
        image[mask] = edge_color
        return image

    return edges
