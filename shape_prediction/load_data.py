import os
import numpy as np
from tiger.io import read_image, write_image
from tiger.patches import PatchExtractor3D
from tqdm.auto import tqdm
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure, binary_opening


class DatasetMask(torch.utils.data.Dataset):
    def __init__(self, msks, patch_size):
        self.original_masks = []
        self.corrupted_masks = []

        for msk in msks:
            m, z = get_sample_mask(msk, patch_size=patch_size)
            self.original_masks.append(m)
            self.corrupted_masks.append(z)

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.orig_masks)

    def __getitem__(self, ind):
        """"
        Return a single sample:
        """
        i = torch.tensor(self.original_masks[ind])
        m = torch.tensor(self.corrupted_masks[ind])
        return i, m


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for a simple dataset containing patches of vertebra and associated scores.
    Also keeps track of where the vertebra was found in the dataset (ID and type)'
    TODO: get more than one sample per image: maybe randomly select image and randomly select v
    """
    def __init__(self, imgs, msks, patch_size, free_form=False):
        self.images = []
        self.masks = []
        self.z = []

        for img, msk in zip(imgs, msks):
            i, m, z = get_sample(img, msk, patch_size=patch_size, free_form=free_form)

            # clip and normalize image
            min, max = -1000, 1500
            i = np.clip(i, min, max)
            i = 2 * (i - min) / (max - min) - 1
            self.images.append(i)
            self.masks.append(m)
            self.z.append(z)

    def __len__(self):
        """
        Returns N, the number of vertebrae in this dataset.
        """
        return len(self.images)

    def __getitem__(self, ind):
        """"
        Return a single sample:
        """
        i = torch.tensor(self.images[ind])
        m = torch.tensor(self.masks[ind])
        z = torch.tensor(self.z[ind])
        stack = torch.stack((i, m, z))
        return stack


def get_sample_mask(msk, patch_size):
    """
    Given an segmentation masks of the spine.
    extracts samples for the in painting task by randomly masking out a vertebra.

    returns:
    m: intact mask
    z: corrupted mask
    """
    # randomly select a vertebra
    verts = np.unique(msk)[1:]
    random_vert = np.random.choice(verts)

    # extract a patch from the mask
    centre = tuple(np.mean(np.argwhere(msk == random_vert), axis=0, dtype=int))
    patch_extracter_msk = PatchExtractor3D(msk)
    m = patch_extracter_msk.extract_cuboid(centre, patch_size)
    z = np.where(m == random_vert, 0, m)
    return m, z


def get_sample(img, msk, patch_size, free_form=False):
    """
    Given an image of a spine and a segmentation masks for the vertebrae
    extracts samples for the in painting task by randomly masking out a vertebra.

    returns:
    i: intact image (patch_size)
    m: mask over one vert (patch_size)
    """
    # randomly select a vertebra
    verts = np.unique(msk)[1:]
    random_vert = np.random.choice(verts)

    # extract a patch from both image and mask
    centre = tuple(np.mean(np.argwhere(msk == random_vert), axis=0, dtype=int))
    patch_extracter_img = PatchExtractor3D(img, pad_value=-1000)
    patch_extracter_msk = PatchExtractor3D(msk)

    i = patch_extracter_img.extract_cuboid(centre, patch_size)
    m = patch_extracter_msk.extract_cuboid(centre, patch_size)

    # construct binary mask, only this vertebra
    m = np.where(m == random_vert, m, 0)

    if free_form:
        # dilate around mask, use structure to smooth so original shape is not given away
        z = binary_opening(m, structure=generate_binary_structure(3, 3), iterations=7)
        z = binary_dilation(z, structure=generate_binary_structure(3, 3), iterations=3)
    else:
        # compute bbox around vert
        x = np.any(m, axis=(0, 1))
        y = np.any(m, axis=(0, 2))
        z = np.any(m, axis=(1, 2))

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        # construct mask for this vert
        m = np.zeros(m.shape)
        m[zmin: zmax, ymin:ymax, xmin:xmax] = 1

    return i, m, z


def load_data(data_dir, cases=100):
    """
    Loads images and masks from a directory (on image level).
    Note: currently loads resampled images.
    """
    img_dir = os.path.join(data_dir, 'images')
    msk_dir = os.path.join(data_dir, 'masks')

    # load all masks
    msk_paths = [os.path.join(msk_dir, f) for f in sorted(os.listdir(msk_dir)) if 'resampled' in f][:cases]
    msk_ids = [f.split('/')[-1].split('.')[0] for f in msk_paths]

    # only load the images that are also present in the masks
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if any(id in f for id in msk_ids)][:cases]
    imgs = np.empty(len(img_paths), dtype=object)
    msks = np.empty(len(msk_paths), dtype=object)

    # load images
    print('Loading images from {}...'.format(img_dir))
    for i, path in enumerate(tqdm(img_paths)):
        img, header = read_image(path)
        img = np.rot90(img, axes=(1, 2))
        imgs[i] = img

    # load masks
    print('Loading masks from {}...'.format(msk_dir))
    for i, path in enumerate(tqdm(msk_paths)):
        msk, header = read_image(path)
        msk = np.rot90(msk, axes=(1, 2))
        msk = np.where(msk > 100, msk - 100, msk)     # if partially visible
        msk = np.where(msk > 7, msk - 7, 0)           # remove c vertebrae, shift labels: 0 => bg, 1-18 => T1-L6
        msks[i] = msk

    return imgs, msks
