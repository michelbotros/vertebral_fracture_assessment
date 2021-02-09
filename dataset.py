import numpy as np
from tiger.patches import PatchExtractor3D
from torch.utils.data import DataLoader
import torch


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for a simple dataset containing masks of vertebrae and associated scores.'
    """
    def __init__(self, scores, masks, patch_size):
        # shape = (N, 2), consisting of a grade and case for each of the 5 vertebrae, N is the number of vertebae
        self.scores = scores.reshape(-1, scores.shape[2])
        self.scores = np.any(self.scores > 0, axis=1).astype(int)  # shape = (N, 1) scores binary
        self.patches = []  # patch containing the mask of this vertebrae

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

