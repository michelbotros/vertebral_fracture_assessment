import os
import numpy as np
from tiger.io import read_image, write_image
from tiger.patches import PatchExtractor3D
from tqdm.auto import tqdm
import torch
import pandas as pd
from monai.transforms import Compose, RandRotate, RandZoom


class DatasetMask(torch.utils.data.Dataset):
    """"
    Dataset of samples (x, y)
    x: the input, multiple vertebrae in diff channels shape = (context=2, patch_size, patch_size, patch_size)
    y: the output, shape to be predicted shape = (1, patch_size, patch_size, patch_size)

    """
    def __init__(self, scores, masks, patch_size, context=2, transforms=False):
        self.x = []                            # the input: multiple vertebrae in diff channels (channel, patch_size, patch_size, patch_size)
        self.y = []                            # the output: predicted shape of the one left out in the middle of the input
        self.transforms = transforms           # whether to apply data aug
        self.context = context                 # how many neighbouring vertebrae to use as input

        for row, mask in enumerate(tqdm(masks)):
            # get the scores for every vert
            vert_scores = scores[row][2:].reshape(18, 2).astype(float)

            # extract all vert patches in the mask
            patches, verts_present = self.extract_vertebrae(mask, patch_size)

            # go over all the scores
            for i, vert_score in enumerate(vert_scores):
                if not (np.isnan(vert_score).any()):
                    grade = vert_score[0]
                    vert = i + 1

                    # for the ones that are healthy and present in the mask => construct sample (x, y)
                    if vert in verts_present and grade == 0:
                        x = np.zeros((context*2, *patch_size))
                        y = np.expand_dims(patches[np.where(verts_present == vert)[0][0]], axis=0)

                        # add neighbours in channels
                        for p, d in enumerate([d for d in range(-context, context + 1) if d != 0]):
                            neighbour = vert + d
                            if neighbour in verts_present:
                                x[p] = patches[np.where(verts_present == neighbour)[0][0]]

                        # add sample
                        self.x.append(x)
                        self.y.append(y)

    def extract_vertebrae(self, mask, patch_size):
        """
        Extracts all patches with patch_size out of a complete mask and filters out other vertebrae.
        Returns: list of patches, containing one vertebra (binary)
                 list of which vertebrae are present
        """
        verts_present = np.unique(mask)[1:]
        patches = []

        for vert in verts_present:   # exclude background
            centre = tuple(np.mean(np.argwhere(mask == vert), axis=0, dtype=int))
            patch_extracter_msk = PatchExtractor3D(mask)
            m = patch_extracter_msk.extract_cuboid(centre, patch_size)
            m = np.where(m == vert, m > 0, 0)  # keep only this vert and make binary
            patches.append(m)

        return patches, verts_present

    def __len__(self):
        """
        Returns N, the number of samples in this dataset.
        """
        return len(self.x)

    def __getitem__(self, i):
        """"
        Return a single sample:
        """
        x = torch.tensor(self.x[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y


def split_train_val_test(msks, scores, patch_size, train_percent=0.8, val_percent=0.1):
    """
    Splits the loaded data into a train, validation and test set.
    """
    # make train and val split on image level
    IDs = np.arange(len(msks))
    np.random.seed(1993)
    np.random.shuffle(IDs)

    N = len(IDs)
    n_train_end = int(train_percent * N)
    n_val_end = int(val_percent * N) + n_train_end
    train_ids = IDs[:n_train_end]
    val_ids = IDs[n_train_end:n_val_end]
    test_ids = IDs[n_val_end:]

    # convert the scores to numpy, for indexing
    np_scores = scores.to_numpy()

    # apply split
    print('Available cases: {} \ntrain: {}, val: {}, test: {}'.format(N, len(train_ids), len(val_ids), len(test_ids)))
    print('Extracting patches...')
    train_set = DatasetMask(np_scores[train_ids], msks[train_ids], patch_size, transforms=True)
    val_set = DatasetMask(np_scores[val_ids], msks[val_ids], patch_size)
    test_set = DatasetMask(np_scores[test_ids], msks[test_ids], patch_size)
    print('train: {}, val: {}, test: {}'.format(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    return train_set, val_set, test_set


def load_masks(data_dir, cases=120):
    """
    Loads masks and scores from a directory.
    """
    msk_dir = os.path.join(data_dir, 'masks_bodies')
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir)) if 'resampled' in f][:cases]
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv')).iloc[:cases]
    msks = np.empty(len(msk_paths), dtype=object)

    # load masks
    print('Loading masks from {}...'.format(msk_dir))
    for i, path in enumerate(tqdm(msk_paths)):
        msk, header = read_image(path)
        msk = np.rot90(msk, axes=(1, 2))
        msk = np.where(msk > 100, 0, msk)             # if partially visible remove
        msk = np.where(msk > 7, msk - 7, 0)           # remove c vertebrae, shift labels: 0 => bg, 1-18 => T1-L6
        msks[i] = msk

    return msks, scores