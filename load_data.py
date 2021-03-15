import numpy as np
import os
from tqdm.auto import tqdm
import pandas as pd
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image, write_image
from tiger.patches import PatchExtractor3D
import torch
from monai.transforms import Compose, RandGaussianNoise, RandRotate, RandGaussianSmooth


class Sampler:
    """"
    Simple sampler that ensures there is at least one positive label but also one negative label in the batch.
    """
    def __init__(self, scores, batch_size):
        self.scores = scores
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
    def __init__(self, scores, images, masks, patch_size, transforms=False, norm_stats=None):
        self.img_patches = []                  # (N, patchsize) the image patch
        self.msk_patches = []                  # (N, patchsize) the mask patch
        self.grades = []                       # (N, 1)         the grade
        self.cases = []                        # (N, 1)         the case
        self.sources = []                      # (N, 1)         dataset of where the image was found
        self.IDS = []                          # (N, 1)         ID of the image, in which this vertebra is found
        self.vertebrae = []                    # (N, 1)         1-18 => T1-L6
        self.patch_size = patch_size
        self.transforms = transforms           # whether to apply transforms or not

        # the patch extraction
        for row, mask in enumerate(masks):
            # get the dataset and id of this case
            source = scores[row][0]
            id = scores[row][1]

            # get the vert scores, 18 vertebrae, grade and case, need float to detect nans
            vert_scores = scores[row][2:].reshape(18, 2).astype(float)

            # find annotated labels in the score sheet
            for i, vert_score in enumerate(vert_scores):
                if not (np.isnan(vert_score).any()):
                    label = i + 1                         # exclude background

                    # if we also find this label in the mask
                    if label in np.unique(mask):
                        centre = tuple(np.mean(np.argwhere(mask == label), axis=0, dtype=int))
                        patch_extracter_img = PatchExtractor3D(images[row], pad_value=-1000)  # pad with air
                        patch_img = patch_extracter_img.extract_cuboid(centre, self.patch_size)
                        patch_img = np.clip(patch_img, -1000, 3000)
                        patch_extracter_msk = PatchExtractor3D(mask)
                        patch_msk = patch_extracter_msk.extract_cuboid(centre, self.patch_size)
                        patch_msk = np.where(patch_msk == label, patch_msk, 0)  # only contain this vertebra, keep label

                        # add score and info about this patch
                        self.img_patches.append(patch_img)
                        self.msk_patches.append(patch_msk)
                        self.grades.append(vert_score[0])   # grade
                        self.cases.append(vert_score[1])    # case
                        self.sources.append(source)
                        self.IDS.append(id)
                        self.vertebrae.append(label)

        # convert to numpy
        self.img_patches = np.asarray(self.img_patches)
        self.msk_patches = np.asarray(self.msk_patches)

        # get norm stats
        if norm_stats:
            self.img_mean, self.img_std, self.msk_mean, self.msk_std = norm_stats
        else:
            self.img_mean, self.img_std = np.mean(self.img_patches), np.std(self.img_patches)
            self.msk_mean, self.msk_std = np.mean(self.msk_patches), np.std(self.msk_patches)

        # apply normalization => zero mean, unit variance
        self.img_patches = (self.img_patches - self.img_mean) / self.img_std
        self.msk_patches = (self.msk_patches - self.msk_mean) / self.msk_std

    def norm_stats(self):
        return self.img_mean, self.img_std, self.msk_mean, self.msk_std

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.img_patches)

    def __getitem__(self, i):
        """"
        Return a single sample: a patch of mask containing one vertebra and its binary score"
        Applies data augmentation
        """
        # get image, mask and score
        patch_img = self.img_patches[i]
        patch_msk = self.msk_patches[i]
        y = self.grades[i]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        if self.transforms:
            # define transforms
            spatial_transforms = Compose([RandRotate(prob=0.25, range_x=0.3, range_y=0.3, range_z=0, mode='nearest')])
            other_transforms = Compose([RandGaussianNoise(prob=0.25, mean=0, std=0.5),
                                        RandGaussianSmooth(prob=0.25, sigma_x=(0, 1.5), sigma_y=(0, 1.5), sigma_z=(0, 1.5))])

            # apply some on just the image, spatial transforms need to be applied to both
            patch_img = other_transforms(patch_img)
            x = spatial_transforms(np.stack((patch_img, patch_msk)))
            x = torch.tensor(x, dtype=torch.float32)
            return x, y

        # stack and to tensor
        x = np.stack((patch_img, patch_msk))
        x = torch.tensor(x, dtype=torch.float32)
        return x, y


def split_train_val_test(imgs, msks, scores, patch_size, data_aug, train_percent=0.8, val_percent=0.1):
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
    train_set = Dataset(np_scores[train_ids], imgs[train_ids], msks[train_ids], patch_size, transforms=data_aug)

    # normalize test and val set with stats from train set
    norm_stats = train_set.norm_stats()
    val_set = Dataset(np_scores[val_ids], imgs[val_ids], msks[val_ids], patch_size, transforms=False, norm_stats=norm_stats)
    test_set = Dataset(np_scores[test_ids], imgs[test_ids], msks[test_ids], patch_size, transforms=False, norm_stats=norm_stats)
    print('train: {}, val: {}, test: {}'.format(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    return train_set, val_set, test_set


def load_data(data_dir):
    """"
    Loads images, masks and scores from a directory (on image level).
    Note: currently loads resampled images.
    """
    img_dir = os.path.join(data_dir, 'images')
    msk_dir = os.path.join(data_dir, 'masks')

    # load all masks
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir)) if 'resampled' in f]
    msk_ids = [f.split('/')[-1].split('.')[0] for f in msk_paths]

    # only load the images that are also present in the masks
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if any(id in f for id in msk_ids)]

    imgs = np.empty(len(img_paths), dtype=object)
    msks = np.empty(len(msk_paths), dtype=object)
    scores = pd.read_csv(os.path.join(data_dir, 'scores.csv'))

    # load images
    print('Loading images from {}...'.format(img_dir))
    for i, path in enumerate(img_paths):
        img, header = read_image(path)
        img = np.rot90(img, axes=(1, 2))
        imgs[i] = img

    # load masks
    print('Loading masks from {}...'.format(msk_dir))
    for i, path in enumerate(msk_paths):
        msk, header = read_image(path)
        msk = np.rot90(msk, axes=(1, 2))
        msk = np.where(msk > 7, msk - 7, 0)          # remove c vertebrae, shift labels: 0 => bg, 1-18 => L1-L6
        msks[i] = msk

    return imgs, msks, scores


