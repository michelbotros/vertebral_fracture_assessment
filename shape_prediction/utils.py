import numpy as np
import math
from scipy.ndimage import rotate
from sklearn.linear_model import LogisticRegression
from tiger.patches import PatchExtractor3D
import math


def extract_vertebrae(mask, patch_size):
    """
    Extracts all patches with patch_size out of a complete mask and filters out other vertebrae.
    Returns: list of patches, containing one vertebra (binary)
             list of which vertebrae are present
             list of orientations of the vertebrae
    """
    verts_present = np.unique(mask)[1:]
    patches = []
    centres = []

    # get patches and centres
    for vert in verts_present:  # exclude background
        centre = tuple(np.mean(np.argwhere(mask == vert), axis=0, dtype=int))
        patch_extracter_msk = PatchExtractor3D(mask)
        m = patch_extracter_msk.extract_cuboid(centre, patch_size)
        m = np.where(m == vert, m > 0, 0)  # keep only this vert and make binary
        patches.append(m)
        centres.append(centre)

    # compute orientations
    orients = []

    for i, vert in enumerate(verts_present):

        if i == 0:
            a_x, a_y = centres[i][2], centres[i][1]  # this and next centre to compute orient
            b_x, b_y = centres[i + 1][2], centres[i + 1][1]
        elif i == len(verts_present) - 1:  # previous and this centre to compute orient
            a_x, a_y = centres[i - 1][2], centres[i - 1][1]
            b_x, b_y = centres[i][2], centres[i][1]
        else:  # previous and next centre to compute orient
            a_x, a_y = centres[i - 1][2], centres[i - 1][1]
            b_x, b_y = centres[i + 1][2], centres[i + 1][1]

        radians = math.atan2(a_y - b_y, a_x - b_x)
        angle = radians * (180 / np.pi) + 90
        orients.append(angle)

    return patches, verts_present, orients


def compute_vertebral_width(mask, orientation):
    """
    Compute the width of a vertebra given its binary mask and it orientation
    """
    # compute the vertebral width at the mid slice
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
    vertebral_width = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    return vertebral_width


def compute_weight_mask(vertebral_width, orientation, c=0.25, size=64):
    """"
    Computes a 2D weight mask given the width of a vertebra that focuses on the centre of the vertebral endplates.
    """
    # construct centre region
    mid = size // 2
    c = int(vertebral_width * c) // 2
    weight_mask = np.zeros(size)
    weight_mask[mid - c: mid + c] = 1
    weight_mask[mid - 4*c: mid - c] = np.linspace(0, 1, num=3*c) ** 2
    weight_mask[mid + c: mid + 4*c] = np.linspace(1, 0, num=3*c) ** 2

    # expand to 2D
    weight_mask = np.broadcast_to(weight_mask, (size, size))

    # rotate the weight mask according to orientation of the vertebra
    weight_mask = rotate(weight_mask, angle=-orientation, axes=(0, 1), reshape=False, mode='nearest', order=1)

    return weight_mask


def compute_abnormality_score(true, pred, orientation, c=0.25, cut_off=0.005):
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
    # log loss like in BCE but only 1 part of it
    abnormality_map = -np.log((1 - pred + 1e-15)) * (1 - true)

    # compute vertebral width
    vert_width = compute_vertebral_width(true, orientation)

    # compute weight mask based on the width and orientation of the vertebra
    weight_mask = compute_weight_mask(vert_width, orientation)

    # apply the weight mask
    abnormality_map_weighted = abnormality_map * weight_mask

    # focus on mid in z-axis as well
    mid = abnormality_map_weighted.shape[0] // 2
    centre_slices = int(abnormality_map_weighted.shape[0] * c) // 2
    abnormality_map_weighted = abnormality_map_weighted[mid - centre_slices: mid + centre_slices]

    # aggregate and normalize with vertebral width
    abnormality_score = np.mean(abnormality_map_weighted) / vert_width

    # scale to [0, 1]
    abnormality_score = np.clip(abnormality_score / cut_off, 0, 1)

    return abnormality_score


def compute_grade(abnormality_score):
    """"
    Computes the grade from an abnormality score.
    Thresholds were found using Logistic Regression, fit on the train set.
    """
    thresholds = [0.06954807, 0.13264613, 0.56281656]

    if abnormality_score > thresholds[2]:
        grade = 3
    elif abnormality_score > thresholds[1]:
        grade = 2
    elif abnormality_score > thresholds[0]:
        grade = 1
    else:
        grade = 0

    return grade
