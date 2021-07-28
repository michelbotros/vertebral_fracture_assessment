import numpy as np
import torch
from tiger.resampling import resample_mask
from utils import compute_abnormality_score, compute_grade, extract_vertebrae
from unet import UNet
from load_data import DatasetMask
from config import patch_size, resolution, context
from config import coarse_model_path, refine_model_path
import pandas as pd


class AbnormalityDetectionPipeline:
    def __init__(self):

        # load config
        self.patch_size = patch_size
        self.resolution = resolution
        self.context = context

        # set device to use for this pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load pretrained coarse net
        self.coarse_net = UNet().to(self.device)
        self.coarse_net.load_state_dict(torch.load(coarse_model_path))
        self.coarse_net.eval()

        # load pretrained refine net
        self.refine_net = UNet(in_channels=1).to(self.device)
        self.refine_net.load_state_dict(torch.load(refine_model_path))
        self.refine_net.eval()

    def __call__(self, name, mask, header):

        # resample the mask to target resolution
        print('Resampling to standard resolution.')
        mask = resample_mask(mask, header.spacing, self.resolution)

        # rotate and remove cervical vertebrae and partially visible
        print('Removing partially visible and cervical vertebrae.')
        mask = np.rot90(mask, axes=(1, 2))
        mask = np.where(mask > 100, 0, mask)         # if partially visible remove

        # extract patches around the vertebrae
        print('Extracting vertebrae...')
        patches, verts_present, orients = extract_vertebrae(mask, self.patch_size)
        print('Found {} vertebrae'.format(len(verts_present)))

        # store results
        grades = []
        abnormality_scores = []

        for vert, orientation in zip(verts_present, orients):
            # print('\nAssessing vertebra: {}'.format(vert))

            # construct sample (x, y)
            x = np.zeros((self.context * 2, *patch_size))
            y = patches[np.where(verts_present == vert)[0][0]]

            for p, d in enumerate([d for d in range(-self.context, self.context + 1) if d != 0]):
                neighbour = vert + d
                if neighbour in verts_present:
                    x[p] = patches[np.where(verts_present == neighbour)[0][0]]

            # to tensor and same device as networks
            x = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)

            # make coarse and fine prediction from context
            coarse = self.coarse_net.forward(x)
            fine = self.refine_net.forward(coarse)

            y_pred = torch.sigmoid(fine).detach().cpu().squeeze().numpy()
            abnormality_score = compute_abnormality_score(y, y_pred, orientation)
            # print('Abnormality score: {}'.format(abnormality_score))
            abnormality_scores.append(abnormality_score)
            grade = compute_grade(abnormality_score)
            # print('Grade: {}'.format(grade))
            grades.append(grade)

        # store result in dataframe
        results = pd.DataFrame({'name': name, 'vert': verts_present, 'grade': grades, 'abnormality:': abnormality_scores})

        return results

