import numpy as np
import torch
import torch.nn.functional as F
from tiger.resampling import resample_image, resample_mask
from config import patch_size, resolution
from models.pl_base import Net
from config import model_path, lr, weight_decay
from utils import extract_vertebrae


class GenantClassifierPipeline:
    def __init__(self):

        # load config
        self.patch_size = np.array(patch_size)
        self.resolution = resolution

        # set device to use for this pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load pretrained net
        self.net = Net(lr=lr, weight_decay=weight_decay).to(self.device)
        self.net = self.net.load_from_checkpoint(model_path, lr=lr, weight_decay=weight_decay).eval()

    def __call__(self, image, mask, header):

        print('Resampling to standard resolution.')
        image = resample_image(image, header.spacing, self.resolution)
        mask = resample_mask(mask, header.spacing, self.resolution)

        print('Removing partially visible and cervical vertebrae.')
        mask = np.rot90(mask, axes=(1, 2))
        mask = np.where(mask > 100, 0, mask)         # if partially visible remove
        mask = np.where(mask > 7, mask - 7, 0)       # remove c vertebrae, shift labels: 0 => bg, 1-18 => T1-L6

        print('Extracting vertebrae...')
        patches, verts_present = extract_vertebrae(image, mask, self.patch_size)
        print('Found {} vertebrae'.format(len(verts_present)))

        results = {}

        for vert, patch in zip(verts_present, patches):

            vert = vert + 7
            print('\nAssessing vertebra: {}'.format(vert))

            with torch.no_grad():
                logits_g, logits_c = self.net.forward(patch)
                g_hat = np.argmax(F.log_softmax(logits_g, dim=1).cpu(), axis=1).item()
                c_hat = np.argmax(F.log_softmax(logits_c, dim=1).cpu(), axis=1).item()
                print('Grade: {}, case: {}'.format(g_hat, c_hat))
                results.update({vert: {'grade': g_hat, 'case': c_hat}})

        return results


