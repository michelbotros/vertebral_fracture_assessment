import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image
from tiger.patches import PatchExtractor3D
import torch


class Sampler:
    """"
    Simple sampler that ensures there is at least one positive label but also one negative label in the batch.
    """
    def __init__(self, scores, batch_size):
        self.scores = np.asarray(scores)                            # numpy indexing
        self.batch_size = batch_size
        self.n_batches = len(scores) // self.batch_size             # drop_last = True

    def __iter__(self):
        res = []
        for _ in range(self.n_batches):
            bad_batch = True
            while bad_batch:
                indexes = np.random.choice(len(self.scores), self.batch_size, replace=False)
                bad_batch = not(1 in self.scores[indexes] and 0 in self.scores[indexes])
            res.append(indexes)
        return iter(res)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for a simple dataset containing patches of vertebrae and associated scores.
    Also keeps track of where the vertebrae was found in the dataset (ID and type)'
    """
    def __init__(self, scores, masks, patch_size):
        self.patches = []                      # (N, 1)    N is the number of vertebrae
        self.scores = []                       # (N, 1)    fractured or not
        self.datasets = []                     # (N, 1)    dataset of where the image was found
        self.IDS = []                          # (N, 1)    ID of the image, in which this vertebrae is found
        self.vertebraes = []                   # (N, 1)    8-25: T1-T12, L1-L6

        for row, mask in enumerate(masks):
            # get the dataset and id of this case
            dataset = scores[row][0]
            id = scores[row][1]

            # get the vert scores, 18 vertebraes, grade and case, need float to detect nans
            vert_scores = scores[row][2:].reshape(18, 2).astype(float)

            # find annotated labels in the score sheet
            for i, vert_score in enumerate(vert_scores):
                if not (np.isnan(vert_score).any()):
                    label = i + 8                              # because we skip the 7 C-vertebrae

                    # if we also find this label in the mask
                    if label in np.unique(mask):
                        # get the patch containing this vertebrae
                        centre = np.mean(np.argwhere(mask == label), axis=0, dtype=int)
                        patch_extracter = PatchExtractor3D(mask)
                        patch = patch_extracter.extract_cuboid(centre, patch_size)
                        patch = np.where(patch == label, 1, 0)          # filter patch to only contain this vertebrae

                        # add score and info about this patch
                        self.patches.append(patch)
                        self.scores.append(vert_score.any().astype(int))       # binarize: fractured or not
                        self.datasets.append(dataset)
                        self.IDS.append(id)
                        self.vertebraes.append(label)

        print('Found a total of {} in this set'.format(len(self.patches)))

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

    def fracture_freq(self):
        """"
        Computes the frequency of fractures in the dataset.
        """
        return np.sum(self.scores) / len(self.scores)


def split_train_val(msks, scores, train_val_split, patch_size):
    """
    Splits the loaded data into a train and validation split.
    """
    # make train and val split on image level
    IDs = np.arange(len(msks))
    np.random.shuffle(IDs)

    # choose split
    n_train = int(train_val_split * len(IDs))
    train_IDs = IDs[:n_train]
    val_IDs = IDs[n_train:]

    # convert the scores to numpy, for indexing
    np_scores = scores.to_numpy()

    # apply split
    train_set = Dataset(np_scores[train_IDs], msks[train_IDs], patch_size)
    val_set = Dataset(np_scores[val_IDs], msks[val_IDs], patch_size)

    print('Frequency of fractures in train set: {}'.format(train_set.fracture_freq()))
    print('Frequency of fractures in val set: {}'.format(val_set.fracture_freq()))

    return train_set, val_set, train_IDs, val_IDs


def load_data(data_dir, resolution):
    """"
    Loads images, masks and scores from a directory (on image level).
    """
    img_dir = os.path.join(data_dir, 'images')
    msk_dir = os.path.join(data_dir, 'masks')
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir))]
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv'))

    # load masks
    msks = np.empty(len(msk_paths), dtype=object)
    print('Loading masks from {}...'.format(data_dir))
    for i, path in enumerate(tqdm(msk_paths)):
        mask, header = read_image(path)
        resampled_mask = resample_mask(mask, header.spacing, resolution)
        flipped_msk = np.rot90(resampled_mask, axes=(1, 2))
        msks[i] = flipped_msk

    return msks, scores


