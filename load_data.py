import numpy as np
import os
import pandas as pd
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image
from tiger.patches import PatchExtractor3D
import torch
from monai.transforms import Compose, RandGaussianNoise, RandRotate, RandGaussianSmooth, RandGaussianSharpen


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
    def __init__(self, scores, images, masks, patch_size, transforms=False):
        self.patches = []                      # (N, 2)    N is the number of vertebrae, img and msk channel
        self.scores = []                       # (N, 1)    fractured or not
        self.sources = []                      # (N, 1)    dataset of where the image was found
        self.IDS = []                          # (N, 1)    ID of the image, in which this vertebra is found
        self.vertebrae = []                    # (N, 1)    8-25: T1-T12, L1-L6
        self.transforms = transforms

        # transform after patch extraction
        self.spatial_transforms = Compose([
            RandRotate(range_x=1/6 * np.pi, range_y=1/6 * np.pi, range_z=0, prob=0.3, mode='nearest')
        ])

        self.other_transforms = Compose([
            RandGaussianNoise(prob=0.2),
            RandGaussianSharpen(prob=0.2),
            RandGaussianSmooth(prob=0.2)
        ])

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
                    label = i + 8                              # because we skip the 7 C-vertebrae

                    # if we also find this label in the mask
                    if label in np.unique(mask):
                        # get the patch containing this vertebra
                        centre = tuple(np.mean(np.argwhere(mask == label), axis=0, dtype=int))

                        # patch extractor for the image, pad with -1000 (air)
                        patch_extracter_img = PatchExtractor3D(images[row], pad_value=-1000)
                        patch_img = patch_extracter_img.extract_cuboid(centre, patch_size)

                        # patch extractor for the mask
                        patch_extracter_msk = PatchExtractor3D(mask)
                        patch_msk = patch_extracter_msk.extract_cuboid(centre, patch_size)
                        patch_msk = np.where(patch_msk == label, 1, 0)  # only contain this vertebra, binary

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

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.patches)

    def __getitem__(self, i):
        """"
        Return a single sample: a patch of mask containing one vertebra and its binary score"
        Applies data
        """
        # get patch, consisting of an image and mask
        x = self.patches[i]
        y = self.scores[i]

        # apply transformation, only for the training set
        if self.transforms:
            # apply spatial transform on both image and mask
            x = self.spatial_transforms(x)

            # apply the others only on the image
            x[0] = self.other_transforms(x[0])

        # to tensor
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        x = torch.tensor(x, dtype=torch.float32)
        return x, y

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

    def frac_freq(self):
        """"
        Computes the frequency of fractures in the dataset.
        """
        return np.sum(self.scores) / len(self.scores)


def split_train_val_test(imgs, msks, scores, patch_size, data_aug, train_percent=0.7, val_percent=0.2):
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
    val_set = Dataset(np_scores[val_ids], imgs[val_ids], msks[val_ids], patch_size)
    test_set = Dataset(np_scores[test_ids], imgs[test_ids], msks[test_ids], patch_size)

    print('train: {}, val: {}, test: {}'.format(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('Frequencies of fractures: \ntrain: {}, val: {}, test: {}'.format(train_set.frac_freq(),
                                                                            val_set.frac_freq(),
                                                                            test_set.frac_freq()))
    return train_set, val_set, test_set


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
    for i, path in enumerate(img_paths):
        image, header = read_image(path)
        resampled_image = resample_image(image, header.spacing, resolution)
        flipped_img = np.rot90(resampled_image, axes=(1, 2))
        clipped_img = np.clip(flipped_img, -1000, 2000)
        imgs[i] = clipped_img

    # load masks,
    print('Loading masks from {}...'.format(msk_dir))
    for i, path in enumerate(msk_paths):
        mask, header = read_image(path)
        resampled_mask = resample_mask(mask, header.spacing, resolution)
        flipped_msk = np.rot90(resampled_mask, axes=(1, 2))
        msks[i] = flipped_msk

    return imgs, msks, scores


