import numpy as np
import torch
import os
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image
from config import patch_size, resolution, batch_size, lr, weight_decay
from models.pl_base import Net
from torch.utils.data import DataLoader
from load_data import Dataset
import pandas as pd


class GenantClassifierPipeline:
    def __init__(self, model_path):

        # load config
        self.patch_size = patch_size
        self.resolution = resolution
        self.batch_size = batch_size

        # set device to use for this pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load pretrained net
        self.net = Net(lr=lr, weight_decay=weight_decay).to(self.device)
        self.net.eval()
        self.net = self.net.load_from_checkpoint(model_path, lr=lr, weight_decay=weight_decay)

    def __call__(self, name, image, mask, header):

        print('=> Resampling to standard resolution.')
        image = resample_image(image, header.spacing, self.resolution)
        mask = resample_mask(mask, header.spacing, self.resolution)

        # we trained it with everything rotated :|
        image = np.rot90(image, axes=(1, 2))
        mask = np.rot90(mask, axes=(1, 2))

        print('=> Removing partially visible and cervical vertebrae.')
        mask = np.where(mask > 100, 0, mask)         # if partially visible remove
        mask = np.where(mask > 7, mask - 7, 0)       # remove c vertebrae, shift labels: 0 => bg, 1-18 => T1-L6

        # dummy scores (scores are not used)
        scores = np.zeros((1, 38))
        imgs = np.array([image])
        msks = np.array([mask])

        print('=> Extracting vertebrae...')
        test_set = Dataset(scores, imgs, msks, self.patch_size, transforms=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=2, shuffle=False, drop_last=False)
        verts = [v + 7 for v in test_set.vertebrae]
        print('=> Found {} vertebrae'.format(test_set.__len__()))

        with torch.no_grad():

            outputs = []

            for idx, batch in enumerate(test_loader):
                outputs.append(self.net.test_step(batch, idx))

            # accumulate results from batches
            _, _, g_hat, c_hat = self.net.accumulate_outputs(outputs)
            results = pd.DataFrame({'name': name, 'vert': verts, 'grade': g_hat})

        return results


def result_segmentation(mask, results):
    """"
    To display the predicted results for each vertebra.
    Take the input mask and label:  normal vertebrae 1, mild fractures 2 etc.
    """
    # remove partially visible
    mask = np.where(mask > 100, 0, mask)

    for vert, grade in zip(results['vert'], results['grade']):
        mask = np.where(mask == vert, grade + 1, mask)

    return mask


def main():
    """
    Pipeline script for docker container on Grand-Challenge.
    """

    # location of input/output
    input_dir_img = '/input/images/ct/'
    input_dir_seg = '/input/images/vertebra/'
    input_path_img = [os.path.join(input_dir_img, f) for f in os.listdir(input_dir_img) if 'mha' in f][0]
    input_path_seg = [os.path.join(input_dir_seg, f) for f in os.listdir(input_dir_seg) if 'mha' in f][0]
    name = input_path_img.split('/')[-1].split('.')[0]
    output_path = '/output/results.json'

    # read input (only one set of inputs is expected)
    image, header = read_image(input_path_img)
    mask, _ = read_image(input_path_seg)

    # location of weights
    model_path = '/opt/algorithm/checkpoints/ResNet50_weights.ckpt'

    # make a pipeline
    pipeline = GenantClassifierPipeline(model_path)

    # get results from the pipeline
    results = pipeline(name, image, mask, header)

    # save output
    results.to_json(output_path)


if __name__ == "__main__":
    main()






