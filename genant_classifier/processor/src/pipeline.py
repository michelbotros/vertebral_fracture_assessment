import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image, write_image
from config import patch_size, resolution, batch_size, lr, weight_decay
from models.pl_base import Net
from load_data import Dataset


class GenantClassifierPipeline:
    def __init__(self, model_path):

        # load config
        self.patch_size = patch_size
        self.resolution = resolution
        self.batch_size = batch_size

        print('Configuration:')
        print('Patch size: {}'.format(self.patch_size))
        print('Resolution: {}'.format(self.resolution))
        print('Batch size: {}'.format(self.batch_size))

        # set device to use for this pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load pretrained net
        self.net = Net.load_from_checkpoint(checkpoint_path=model_path, lr=lr, weight_decay=weight_decay).to(self.device)
        self.net.eval()
        print('=> Weights loaded.')

    def __call__(self, image, mask, header):

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
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=0, shuffle=False, drop_last=False)
        verts = [v + 7 for v in test_set.vertebrae]
        print('=> Found {} vertebrae'.format(test_set.__len__()))

        # transform to anatomical labels
        anatomical_labels = ['bg'] + ['C' + str(v) for v in range(1, 8)] + ['T' + str(v) for v in range(1, 13)] + ['L' + str(v) for v in range(1, 7)]
        a_verts = [anatomical_labels[v] for v in verts]

        with torch.no_grad():
            outputs = []

            for idx, batch in enumerate(test_loader):

                # put on device
                batch = [t.to(self.device) for t in batch]
                outputs.append(self.net.test_step(batch, idx))

            # accumulate results from batches
            _, _, g_hat, c_hat = self.net.accumulate_outputs(outputs)
            results = pd.DataFrame({'vert': verts, 'anatomical vert': a_verts, 'grade': g_hat, 'case': c_hat})

        return results


def result_segmentation(mask, results):
    """"
    To display the predicted results for each vertebra.
    Take the input mask and label:  normal vertebrae 1, mild fractures 2 etc.
    """
    # remove partially visible
    mask = np.where(mask > 100, 0, mask)

    for vert, grade in zip(results['vert'], results['grade']):
        mask = np.where(mask == vert, grade + 2, mask)

    return mask


def main():
    """
    Pipeline script for docker container on Grand-Challenge.
    """

    # location of input
    input_dir_img = '/input/images/ct/'
    input_dir_seg = '/input/images/vertebra/'
    input_path_img = [os.path.join(input_dir_img, f) for f in os.listdir(input_dir_img) if 'mha' in f][0]
    input_path_seg = [os.path.join(input_dir_seg, f) for f in os.listdir(input_dir_seg) if 'mha' in f][0]

    # location of output
    file_name = input_path_img.split('/')[-1]
    output_path_seg = os.path.join('/output/images/', file_name)
    output_path_json = '/output/results.json'

    # read input (only one set of inputs is expected)
    image, header = read_image(input_path_img)
    mask, _ = read_image(input_path_seg)

    # location of weights
    model_path = '/opt/algorithm/checkpoints/epoch=18_step=2146_val loss grade=0.34.ckpt'

    # make a pipeline
    pipeline = GenantClassifierPipeline(model_path=model_path)

    # get results from the pipeline
    results = pipeline(image, mask, header)

    # save output, record format
    results.to_json(output_path_json, orient='records')

    # get the result segmentation
    result_seg = result_segmentation(mask=mask, results=results)
    write_image(output_path_seg, result_seg, header)


if __name__ == "__main__":
    main()




