import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image
from tiger.patches import PatchExtractor3D
from torch.utils.data import DataLoader
import torch


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for a simple dataset containing masks of vertebrae and associated scores.'
    """
    def __init__(self, scores, masks, patch_size):
        self.scores = scores.reshape(-1, scores.shape[2])          # shape = (N, 2), grade and case
        self.scores = np.any(self.scores > 0, axis=1).astype(int)  # shape = (N, 1), scores binary
        self.patches = []                                          # patch containing the mask of this vertebrae

        # get patches
        for mask in masks:
            labels = np.unique(mask)[1:]
            centres = [np.mean(np.argwhere(mask == l), axis=0, dtype=int) for l in labels]
            patch_extracter = PatchExtractor3D(mask)

            for label, centre in zip(labels, centres):
                patch = patch_extracter.extract_cuboid(centre, patch_size)
                patch = np.where(patch == label, 1, 0)  # filter patch to only contain this vertebrae
                self.patches.append(patch)

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.patches)

    def __getitem__(self, i):
        """"
        Return a single sample: a patch of mask containing one vertebrae and its binary score"
        """
        # add channel dimension, use float32 as type
        X = torch.tensor(self.patches[i], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.scores[i], dtype=torch.float32).unsqueeze(0)
        return X, y


def load_data(data_dir, resolution, train_val_split, patch_size, batch_size, nr_imgs=15):
    """"
    Function to load the images, masks and scores from a directory.
    Returns train and validation data loaders.
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
    print('Loading images...')
    for path in tqdm(img_paths):
        image, header = read_image(path)
        resampled_img = resample_image(image, header.spacing, resolution)  # resample image
        flipped_img = np.rot90(resampled_img, axes=(1, 2))  # flip the image, so patient is upright
        imgs.append(flipped_img)

    # load masks
    print('Loading masks...')
    for path in tqdm(msk_paths):
        mask, header = read_image(path)
        resampled_mask = resample_mask(mask, header.spacing, resolution)
        flipped_msk = np.rot90(resampled_mask, axes=(1, 2))
        msks.append(flipped_msk)

    # convert to numpy for indexing
    imgs = np.asarray(imgs, dtype=object)
    msks = np.asarray(msks, dtype=object)

    # shuffle
    IDs = np.arange(len(msks))
    np.random.shuffle(IDs)

    # choose split
    n_train = int(train_val_split * len(IDs))
    train_IDs = IDs[:n_train]
    val_IDs = IDs[n_train:]

    # apply split
    train_set = Dataset(scores[train_IDs], msks[train_IDs], patch_size)
    val_set = Dataset(scores[val_IDs], msks[val_IDs], patch_size)

    # initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print('Size train set: {}'.format(train_set.__len__()))
    print('Size val set: {}'.format(val_set.__len__()))

    # we can look at rate of which fractures occur in the data set
    train_frac_freq = np.sum(train_set.scores) / len(train_set.scores)
    val_frac_freq = np.sum(val_set.scores) / len(val_set.scores)

    print('Frequency of fractures in train set: {}'.format(train_frac_freq))
    print('Frequency of fractures in val set: {}'.format(val_frac_freq))

    return train_loader, val_loader