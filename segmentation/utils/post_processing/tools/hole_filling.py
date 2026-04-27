import cv2
import numpy as np


def fill_hole_and_denoise(mask, threshold=5):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros_like(mask, np.uint8)
    for i in range(len(contours)):
        tmp_canvas = np.zeros_like(mask, np.uint8)
        tmp_canvas = cv2.drawContours(tmp_canvas, contours, i, 255, -1)
        if np.count_nonzero(tmp_canvas) >= threshold:
            canvas = np.logical_or(canvas, tmp_canvas)

    return canvas

