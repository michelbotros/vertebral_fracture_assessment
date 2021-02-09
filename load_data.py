import os
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image


def load_data(data_dir, resolution, nr_imgs):
    """"
    Function to load the images, masks and scores from a directory.
    """
    img_dir = os.path.join(data_dir, 'images')
    msk_dir = os.path.join(data_dir, 'masks')
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))][:nr_imgs]
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir))][:nr_imgs]
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv'), header=None).to_numpy().reshape(15, 5, 2)[:nr_imgs]

    imgs = []
    msks = []
    hdrs = []

    # load images with headers and resample to working resolution
    print('Loading images ...')
    for path in tqdm(img_paths):
        image, header = read_image(path)
        resampled_img = resample_image(image, header.spacing, resolution)  # resample image
        flipped_img = np.rot90(resampled_img, axes=(1, 2))  # flip the image, so patient is upright
        imgs.append(flipped_img)

    # load masks
    print('Loading masks ...')
    for path in tqdm(msk_paths):
        mask, header = read_image(path)
        resampled_mask = resample_mask(mask, header.spacing, resolution)
        flipped_msk = np.rot90(resampled_mask, axes=(1, 2))
        msks.append(flipped_msk)

    return np.asarray(imgs, dtype=object), np.asarray(msks, dtype=object), scores