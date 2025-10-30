import cv2
import pydicom
import numpy as np
from matplotlib import pyplot as plt

def global_thresholding(img, threshold=None):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if threshold is None:
        T, seg = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        T, seg = cv2.threshold(img_norm, threshold, 255, cv2.THRESH_BINARY)

    return seg, T
def otsu_algo(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hist, _ = np.histogram(img, bins=256, range=(0,256))
    prob = hist / hist.sum()

    score = []
    thresholds = range(256)

    for t in thresholds:
        p1 = np.sum(prob[:t+1])
        p2 = np.sum(prob[t+1:])
        if p1 == 0 or p2 == 0:
            score.append(0)
            continue

        arr_1 = prob[:t+1]
        arr_2 = prob[t+1:]
        indices_1 = np.arange(0, t+1)
        indices_2 = np.arange(t+1, 256)

        m1 = np.sum(arr_1 * indices_1) / p1
        m2 = np.sum(arr_2 * indices_2) / p2

        c_mean = np.sum(prob * np.arange(256))
        g_var = np.sum(prob * (np.arange(256) - c_mean)**2)

        c_var = p1 * p2 * (m1 - m2)**2
        s = c_var / g_var if g_var != 0 else 0
        score.append(s)

    k = np.argmax(score)
    seg = np.zeros_like(img, dtype=np.uint8)
    seg[img > k] = 255
    return seg, k

def connected_component_binary(x,connect):
    mat = (x > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mat, connectivity=connect)
    return num_labels, labels, stats, centroids











