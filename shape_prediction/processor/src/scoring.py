import numpy as np
import math
from scipy.ndimage import rotate
from sklearn.linear_model import LogisticRegression
from tiger.patches import PatchExtractor3D
from scipy.stats import multivariate_normal


def compute_l_lr(mask):
    """
    Compute the left-to-right length.
    """
    mask = np.swapaxes(mask, 0, 2)
    return compute_l_ap(mask, orientation=0)


def compute_l_ap(mask, orientation):
    """
    Compute the anterior-to-posterior length.
    """
    # at the mid slice
    mid = mask.shape[-1] // 2
    mask = mask[mid]

    # construct a line with the orientation given
    line_map = np.zeros_like(mask)
    line_map[mid] = 1
    line_map = rotate(line_map, angle=-orientation, axes=(0, 1), reshape=False, mode='nearest', order=1)

    # check where the vertebra ends
    line_map = line_map & mask
    indices = np.argwhere(line_map)

    # compute outer points
    min_y, max_y = np.min(indices[:, 0]), np.max(indices[:, 0])
    min_x, max_x = np.min(indices[:, 1]), np.max(indices[:, 1])

    # compute distance between points
    l_ap = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    return l_ap


def compute_weight_mask(l_ap, orientation, c=0.25, size=64):
    """"
    Computes a 3D weight mask.
    On the y-axis the mid 1/4 of the anterior to posterior length is considered the center.
    Outside the center a quadratic decrease is used till the anterior and posterior wall.
    On the x-axis the mid 1/4 of the x-axis is considered the center.

    l_ap: anterior-to-posterior length
    orientation: the orientation of the vertebra
    c: ratio which is considered the center
    """
    # construct centre region (y-axis)
    mid = size // 2
    y = int(l_ap * c) // 2

    weight_mask = np.zeros(size)
    weight_mask[mid - y: mid + y] = 1
    weight_mask[mid - 4 * y: mid - y] = np.abs(np.linspace(0, 1, num=3 * y)) ** 2
    weight_mask[mid + y: mid + 4 * y] = np.abs(np.linspace(1, 0, num=3 * y)) ** 2

    # expand to 3D
    weight_mask = np.broadcast_to(weight_mask, (size, size, size)).copy()
    weight_mask = np.swapaxes(weight_mask, 0, 1)

    # remove left and ride side of vertebra (x-axis)
    x = int(size * c) // 2
    weight_mask[:mid - x] = 0
    weight_mask[mid + x:] = 0

    # rotate the weight mask according to orientation of the vertebra
    weight_mask = rotate(weight_mask, angle=-orientation, axes=(1, 2), reshape=False, mode='nearest', order=1)

    return weight_mask


def compute_gaussian_weight_mask(l_ap, l_lr, orientation, size=64):
    """"
    Computes a 3D Gaussian weight mask to exclude differences at the left, right, anterior and posterior side.
    Sigma is chosen so that l_ap = 3 * sigma (excludes the sides)

    l_ap: anterior-to-posterior length
    l_lr: left-to-right length
    orientation: the orientation of the vertebra
    """
    mid = size // 2

    # define 64x64 grid
    x, y = np.mgrid[0:size:(size * 1j), 0:size:(size * 1j)]

    # (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    # construct 2D Gaussian based on the shape of the vert
    mu = np.array([mid, mid])                                 # mean is at center of the vertebral body
    sigma = np.array([l_lr / 6, l_ap / 6])                    # such that l_ap = 3 * sigma
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    # reshape back to a (64, 64) grid
    z = z.reshape(x.shape)

    # scale to have a max of 1
    z = z / np.max(z)
    weight_mask = np.swapaxes(np.broadcast_to(z, (size, size, size)), 0, 1)

    # rotate the weight mask according to orientation of the vertebra
    weight_mask = rotate(weight_mask, angle=-orientation, axes=(1, 2), reshape=False, mode='nearest', order=1)

    return weight_mask


def compute_abnormality_score(true, pred, orientation, cut_off=1.5):
    """
    Computes an abnormality score by comparing the predicted shape of the vertebra with the true shape of the vertebra.
    With focus on the centre of the vertebra.

    true: the real shape of the vert (binary mask)
    pred: the predicted shape of the vert (output of sigmoid)
    orientation: orientation of the true vertebra (computed from centres)
    c: centre region, beyond that region the abnormality is weighed less
    cut_off: upper bound for abnormality score

    returns: abnormality score in  range [0, 1]
    """
    # subtract and get rid of the negative part
    abnormality_map = np.where((pred - true) < 0, 0, pred - true)

    # compute l_ap, l_lr
    l_ap = compute_l_ap(true, orientation)
    l_lr = compute_l_lr(true)

    # compute weight mask based on l_ap, l_lr and orientation of the vertebra
    weight_mask = compute_gaussian_weight_mask(l_ap, l_lr, orientation)

    # apply the weight mask
    abnormality_map_weighted = abnormality_map * weight_mask

    # aggregate and scale with vertebra size
    abnormality_score = np.sum(abnormality_map_weighted) / (l_ap * l_lr)

    # compute grade before scaling
    grade = compute_grade(abnormality_score)

    # scale to [0, 1]
    abnormality_score = np.clip(abnormality_score / cut_off, 0, 1)

    return abnormality_score, grade


def compute_grade(abnormality_score):
    """"
    Computes the grade from an abnormality score.
    Thresholds were found using Logistic Regression, fit on the train set.
    """
    # [0.20195353, 0.27162716, 0.64586459]
    thresholds = [0.30293029, 0.40744074, 0.96879688]

    if abnormality_score > thresholds[2]:
        grade = 3
    elif abnormality_score > thresholds[1]:
        grade = 2
    elif abnormality_score > thresholds[0]:
        grade = 1
    else:
        grade = 0

    return grade
