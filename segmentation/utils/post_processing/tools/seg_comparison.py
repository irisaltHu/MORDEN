from segmentation.utils.metrics import SegmentationMetrics
import os
import numpy as np
import cv2
import torch


def segmentation_compare(label_dir, prediction_dir, image_size=(1024, 1024)):
    """
    :param label_dir: requires str. directory of the labels
    :param prediction_dir: requires str. directory of the predictions
    :param image_size: requires tuple. image size
    :return: None. output the metrics directly
    """
    metrics = SegmentationMetrics(2, 0)

    names = os.listdir(label_dir)
    iou = 0.0
    f1_score = 0.0
    recall = 0.0
    precision = 0.0
    for name in names:
        label = cv2.imread(label_dir + name, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, image_size)
        label = label / np.max(label)
        prediction = cv2.imread(prediction_dir + name, cv2.IMREAD_GRAYSCALE)
        prediction = prediction / np.max(prediction)
        result = metrics.pixel_metrics(torch.stack([torch.from_numpy(prediction)]),
                                       torch.stack([torch.from_numpy(label)]))
        iou += result['iou']['mean']
        f1_score += result['f1-score']['mean']
        recall += result['recall']['mean']
        precision += result['precision']['mean']

    cnt = len(names)
    print(iou / cnt)
    print(f1_score / cnt)
    print(precision / cnt)
    print(recall / cnt)
