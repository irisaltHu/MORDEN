import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def crf_postprocess(image, output, n_iters=5, alpha=8, beta=8, gamma=3):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = image.shape[:2]

    if len(output.shape) == 2:
        prob_map = np.zeros((2, h, w), dtype=np.float32)
        output[output >= 0.5] = 1.0
        output[output < 0.5] = 0.0
        prob_map[0] = 1 - output
        prob_map[1] = output
    else:
        prob_map = output
    n_classes = prob_map.shape[0]

    d = dcrf.DenseCRF2D(w, h, n_classes)

    u = unary_from_softmax(prob_map)
    d.setUnaryEnergy(u.reshape(n_classes, -1).astype(np.float32))

    d.addPairwiseGaussian(sxy=gamma, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(alpha, beta, image.astype(np.uint8), 40, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    q = d.inference(n_iters)
    q = np.array(q)
    res = q[1].reshape(h, w)

    return res

