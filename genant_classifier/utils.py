import numpy as np
import math
from scipy.ndimage import rotate
from sklearn.linear_model import LogisticRegression
from tiger.patches import PatchExtractor3D
import math
from monai.transforms import Compose, RandSpatialCrop
import torch


def extract_vertebrae(image, mask, patch_size):
    """
    Extracts all patches with patch_size out of a complete mask and filters out other vertebrae.
    Returns: list of patches, containing one vertebra (model ready input)
             list of which vertebrae are present
    """
    verts_present = np.unique(mask)[1:]
    patches = []

    # get patches
    for vert in verts_present:  # exclude background

        centre = tuple(np.mean(np.argwhere(mask == vert), axis=0, dtype=int))

        # extract image patch, pad with air, clip between -1000 and 1500
        patch_extracter_img = PatchExtractor3D(image, pad_value=-1000)
        patch_img = patch_extracter_img.extract_cuboid(centre, patch_size + 16)
        print("Shape patch img: {}".format(patch_img.shape))
        patch_img = np.clip(patch_img, -1000, 1500)

        # extract mask patch, keep only this vert
        patch_extracter_msk = PatchExtractor3D(mask)
        patch_msk = patch_extracter_msk.extract_cuboid(centre, patch_size + 16)
        print("Shape patch msk: {}".format(patch_msk.shape))
        patch_msk = np.where(patch_msk == vert, patch_msk, 0)

        # apply normalization
        patch_img = (patch_img + 1000) / 2500
        patch_msk = patch_msk / 18

        # stack image and mask patch
        crop_transform = Compose([RandSpatialCrop(roi_size=patch_size, random_size=False)])
        patch = crop_transform(np.stack((patch_img, patch_msk)))
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

        # add to the lists
        patches.append(patch)

    return patches, verts_present

