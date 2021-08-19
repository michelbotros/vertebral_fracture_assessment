import numpy as np
import torch
import os
import pandas as pd
from tiger.io import read_image, write_image
from tiger.resampling import resample_mask
from utils import compute_abnormality_score, compute_grade, extract_vertebrae
from unet import UNet
from load_data import DatasetMask
from config import patch_size, resolution, context


class AbnormalityDetectionPipeline:
    def __init__(self, coarse_model_path, refine_model_path):

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
        print('=> Loaded weights.')

    def __call__(self, mask, header):

        # resample the mask to target resolution
        print('=> Resampling to standard resolution.')
        mask = resample_mask(mask, header.spacing, self.resolution)

        # rotate and remove cervical vertebrae and partially visible
        print('=> Removing partially visible and cervical vertebrae.')
        mask = np.rot90(mask, axes=(1, 2))
        mask = np.where(mask > 100, 0, mask)                   # if partially visible remove
        mask = np.where((mask > 0) & (mask < 8), 0, mask)      # remove c vertebrae

        # extract patches around the vertebrae
        print('=> Extracting vertebrae...')
        patches, verts_present, orients = extract_vertebrae(mask, self.patch_size)
        print('=> Found {} vertebrae'.format(len(verts_present)))

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

        # transform to anatomical labels
        anatomical_labels = ['bg'] + ['C' + str(v) for v in range(1, 8)] + ['T' + str(v) for v in range(1, 13)] + ['L' + str(v) for v in range(1, 7)]
        a_verts = [anatomical_labels[v] for v in verts_present]

        # store result in dataframe
        results = pd.DataFrame({'vert': verts_present, 'anatomical vert': a_verts, 'grade': grades, 'abnormality': abnormality_scores})

        return results


def result_segmentation(mask, results):
    """"
    To display the predicted results for each vertebra.
    Shows the abnormality score.
    """
    # remove partially visible and C vertebra
    mask = np.where(mask > 100, 0, mask)
    mask = np.where((mask > 0) & (mask < 8), 0, mask)

    # fill in the abnormality score
    for vert, abnormality in zip(results['vert'], results['abnormality']):

        # scale a bit different for the plot (traffic light on Grand-challenge)
        abnormality_plot = np.clip(abnormality * 3, 0, 1)
        mask = np.where(mask == vert, int(abnormality_plot * 255), mask)

    return mask


def main():
    """
    Pipeline script for docker container on Grand-Challenge.
    """

    # location of input
    input_dir_img = '/input/images/ct/'
    input_dir_seg = '/input/images/vertebral-body/'
    input_path_img = [os.path.join(input_dir_img, f) for f in os.listdir(input_dir_img) if 'mha' in f][0]
    input_path_seg = [os.path.join(input_dir_seg, f) for f in os.listdir(input_dir_seg) if 'mha' in f][0]

    # location of output
    file_name = input_path_img.split('/')[-1]
    output_path_seg = os.path.join('/output/images/', file_name)
    output_path_json = '/output/results.json'

    # read input
    mask, header = read_image(input_path_seg)

    # location of weights
    course_model_path = '/opt/algorithm/checkpoints/best_model_epoch_41_loss_0.032.pt'
    refine_model_path = '/opt/algorithm/checkpoints/best_model_epoch_191_loss_0.033.pt'

    # make a pipeline
    pipeline = AbnormalityDetectionPipeline(course_model_path, refine_model_path)

    # get results from the pipeline
    results = pipeline(mask, header)

    # save output, record format
    results.to_json(output_path_json, orient='records')

    # get the result segmentation
    result_seg = result_segmentation(mask=mask, results=results)
    write_image(output_path_seg, result_seg, header)


if __name__ == "__main__":
    main()
