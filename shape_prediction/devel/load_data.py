import os
import numpy as np
from tiger.io import read_image, write_image
from tqdm.auto import tqdm
import torch
import pandas as pd
from monai.transforms import Compose, RandRotate, RandZoom
from utils import extract_vertebrae


class DatasetMask(torch.utils.data.Dataset):
    """"
    Dataset of samples (x, y)
    x: the input, multiple vertebrae in diff channels shape = (context=2, patch_size, patch_size, patch_size)
    y: the output, shape to be predicted shape = (1, patch_size, patch_size, patch_size)
    by default loads only the healthy vertebrae
    """
    def __init__(self, scores, masks, patch_size, healthy_only=True, context=2):
        self.x = []                            # the input: multiple vertebrae in diff channels (channel, patch_size, patch_size, patch_size)
        self.y = []                            # the output: predicted shape of the one left out in the middle of the input
        self.g = []                            # the grade of this sample
        self.c = []                            # the case of this sample
        self.context = context                 # how many neighbouring vertebrae to use as input
        self.max_grade = 0 if healthy_only else 3
        self.IDS = []
        self.orientation = []

        for row, mask in enumerate(tqdm(masks)):
            id = scores[row][1]

            # get the scores for every vert
            vert_scores = scores[row][2:].reshape(18, 2).astype(float)

            # extract all vert patches in the mask
            patches, verts_present, orients = extract_vertebrae(mask, patch_size)

            # go over all the scores
            for i, vert_score in enumerate(vert_scores):
                if not (np.isnan(vert_score).any()):
                    grade, case = vert_score[0], vert_score[1]
                    vert = i + 1

                    # for the ones that are healthy and present in the mask => construct sample (x, y)
                    if vert in verts_present and grade <= self.max_grade:
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
                        self.g.append(grade)
                        self.c.append(case)
                        self.IDS.append((id, vert))
                        self.orientation.append(orients[np.where(verts_present == vert)[0][0]])

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
        g = torch.tensor(self.g[i], dtype=torch.float32)
        return x, y, g


def split_train_val_test(msks, scores, patch_size, train_percent=0.8, val_percent=0.1, healthy_only=True):
    """
    Splits the loaded data into a train, validation and test set.
    """
    # neck images
    rm_ids = [201, 202, 205, 207, 208, 209, 212, 214, 215, 221, 223, 225, 226, 227, 230, 232, 235, 239, 242, 243]

    # indexes to remove
    rm_inds = list(scores[scores['ID'].isin(rm_ids)].index)

    # make train and val split on image level
    IDs = np.arange(len(msks))
    np.random.seed(1993)
    np.random.shuffle(IDs)

    N = len(IDs)
    n_train_end = int(train_percent * N)
    n_val_end = int(val_percent * N) + n_train_end
    train_ids = [x for x in IDs[:n_train_end] if x not in rm_inds]
    val_ids = [x for x in IDs[n_train_end:n_val_end] if x not in rm_inds]
    test_ids = [x for x in IDs[n_val_end:] if x not in rm_inds]

    print('Test IDs: {}'.format(test_ids))

    # convert the scores to numpy, for indexing
    np_scores = scores.to_numpy()

    # apply split
    print('Available cases: {} \ntrain: {}, val: {}, test: {}'.format(N, len(train_ids), len(val_ids), len(test_ids)))
    print('Extracting patches...')
    train_set = DatasetMask(np_scores[train_ids], msks[train_ids], patch_size, healthy_only=healthy_only)
    val_set = DatasetMask(np_scores[val_ids], msks[val_ids], patch_size, healthy_only=healthy_only)
    test_set = DatasetMask(np_scores[test_ids], msks[test_ids], patch_size, healthy_only=False)
    print('train: {}, val: {}, test: {}'.format(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    return train_set, val_set, test_set


def load_masks(data_dir, cases=120):
    """
    Loads masks and scores from a directory.
    """
    # only the clean masks
    msk_clean_dir = os.path.join(data_dir, 'masks_bodies_final')
    msk_clean_paths = [os.path.join(msk_clean_dir, f) for f in sorted(os.listdir(msk_clean_dir)) if 'mha' in f][:cases]

    # get only the scores of cleaned masks
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv'))

    # store masks in array
    msks = np.empty(len(msk_clean_paths), dtype=object)

    # load masks
    print('Loading masks from {}...'.format(msk_clean_dir))
    for i, path in enumerate(tqdm(msk_clean_paths)):
        msk, header = read_image(path)
        msk = np.rot90(msk, axes=(1, 2))
        msk = np.where(msk > 100, 0, msk)             # if partially visible remove
        msk = np.where(msk > 7, msk - 7, 0)           # remove c vertebrae, shift labels: 0 => bg, 1-18 => T1-L6
        msks[i] = msk

    return msks, scores