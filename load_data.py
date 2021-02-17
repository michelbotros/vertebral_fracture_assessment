import numpy as np
from tqdm.auto import tqdm
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
    Dataset class for a simple dataset containing patches of vertebra and associated scores.
    Also keeps track of where the vertebra was found in the dataset (ID and type)'
    """
    def __init__(self, scores, images, masks, patch_size):
        self.patches = []                      # (N, 2)    N is the number of vertebrae, img and msk channel
        self.scores = []                       # (N, 1)    fractured or not
        self.sources = []                      # (N, 1)    dataset of where the image was found
        self.IDS = []                          # (N, 1)    ID of the image, in which this vertebra is found
        self.vertebrae = []                    # (N, 1)    8-25: T1-T12, L1-L6

        for row, mask in enumerate(tqdm(masks)):
            # get the dataset and id of this case
            source = scores[row][0]
            id = scores[row][1]

            # get the vert scores, 18 vertebrae, grade and case, need float to detect nans
            vert_scores = scores[row][2:].reshape(18, 2).astype(float)

            # find annotated labels in the score sheet
            for i, vert_score in enumerate(vert_scores):
                if not (np.isnan(vert_score).any()):
                    label = i + 8                              # because we skip the 7 C-vertebrae

                    # if we also find this label in the mask
                    if label in np.unique(mask):
                        # get the patch containing this vertebra
                        centre = tuple(np.mean(np.argwhere(mask == label), axis=0, dtype=int))

                        # patch extractor for the image
                        patch_extracter_img = PatchExtractor3D(images[row])
                        patch_img = patch_extracter_img.extract_cuboid(centre, patch_size)

                        # patch extractor for the mask
                        patch_extracter_msk = PatchExtractor3D(mask)
                        patch_msk = patch_extracter_msk.extract_cuboid(centre, patch_size)
                        patch_msk = np.where(patch_msk == label, patch_msk, 0)  # only contain this vertebra

                        # add channel dimension
                        patch_img = np.expand_dims(patch_img, axis=0)
                        patch_msk = np.expand_dims(patch_msk, axis=0)
                        patch = np.concatenate((patch_img, patch_msk))

                        # add score and info about this patch
                        self.patches.append(patch)
                        self.scores.append(vert_score.any().astype(int))       # binarize: fractured or not
                        self.sources.append(source)
                        self.IDS.append(id)
                        self.vertebrae.append(label)
                    # else:
                    #     print('Did not find label {} in the mask as well. Case: {} {}'.format(label, source, id))

        print('Found a total of {} annotated vertebrae in this set'.format(len(self.patches)))

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.patches)

    def __getitem__(self, i):
        """"
        Return a single sample: a patch of mask containing one vertebra and its binary score"
        """
        # use float32 as type,
        X = torch.tensor(self.patches[i], dtype=torch.float32)
        y = torch.tensor(self.scores[i], dtype=torch.float32).unsqueeze(0)
        return X, y

    def get_scores(self):
        return np.asarray(self.scores)

    def get_sources(self):
        return np.asarray(self.sources)

    def get_vertebrae(self):
        return np.asarray(self.vertebrae)

    def get_patches(self):
        return np.asarray(self.patches)

    def get_ids(self):
        return np.asarray(self.IDS)

    def fracture_freq(self):
        """"
        Computes the frequency of fractures in the dataset.
        """
        return np.sum(self.scores) / len(self.scores)


def split_train_val(imgs, msks, scores, train_val_split, patch_size):
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

    print('Available cases: {} \nSplitting into train: {}, val: {}'.format(len(msks), n_train, len(msks) - n_train))

    # apply split
    print('Extracting patches train set...')
    train_set = Dataset(np_scores[train_IDs], imgs[train_IDs], msks[train_IDs], patch_size)
    print('Extracting patches val set...')
    val_set = Dataset(np_scores[val_IDs], imgs[val_IDs], msks[val_IDs], patch_size)

    print('Frequency of fractures in train set: {}'.format(train_set.fracture_freq()))
    print('Frequency of fractures in val set: {}'.format(val_set.fracture_freq()))

    return train_set, val_set, train_IDs, val_IDs


def load_data(data_dir, resolution):
    """"
    Loads images, masks and scores from a directory (on image level).
    """
    img_dir = os.path.join(data_dir, 'images')
    msk_dir = os.path.join(data_dir, 'masks')

    # load all masks
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir))]
    msk_ids = [f.split('/')[-1].split('.')[0] for f in msk_paths]

    # only load the images that are also present in the masks
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if any(id in f for id in msk_ids)]

    imgs = np.empty(len(img_paths), dtype=object)
    msks = np.empty(len(msk_paths), dtype=object)
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv'))

    # load images
    print('Loading images from {}...'.format(img_dir))
    for i, path in enumerate(tqdm(img_paths)):
        image, header = read_image(path)
        resampled_image = resample_mask(image, header.spacing, resolution)
        flipped_img = np.rot90(resampled_image, axes=(1, 2))
        imgs[i] = flipped_img

    # load masks
    print('Loading masks from {}...'.format(msk_dir))
    for i, path in enumerate(tqdm(msk_paths)):
        mask, header = read_image(path)
        resampled_mask = resample_mask(mask, header.spacing, resolution)
        flipped_msk = np.rot90(resampled_mask, axes=(1, 2))
        msks[i] = flipped_msk

    return imgs, msks, scores


