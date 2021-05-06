import os
from tiger.resampling import resample_image, resample_mask
from tiger.io import read_image, write_image
from config import nlst_dir, resolution

img_outdir = os.path.join(nlst_dir, 'images')
msk_outdir = os.path.join(nlst_dir, 'masks')

img_indir = os.path.join(nlst_dir, 'images_normal', 'mha', 'images')
msk_indir = os.path.join(nlst_dir, 'masks_normal_processor', 'vertebra_masks')

print('Resampling images from: {}'.format(img_indir))
print('Resampling masks from: {}'.format(msk_indir))

img_paths = [os.path.join(img_indir, f) for f in sorted(os.listdir(img_indir))]
msk_paths = [os.path.join(msk_indir, f) for f in sorted(os.listdir(msk_indir))]

print('Resampling images...')
for path in img_paths:
    id = path.split('/')[-1].split('.mha')[0]
    save_path = os.path.join(img_outdir, 'resampled_' + id + '.mha')
    image, header = read_image(path)
    resampled_img = resample_image(image, header.spacing, resolution)
    header['spacing'] = resolution
    write_image(save_path, resampled_img, header)

print('Resampling masks...')
for path in msk_paths:
    id = path.split('/')[-1].split('.mha')[0]
    save_path = os.path.join(msk_outdir, 'resampled_' + id + '.mha')
    mask, header = read_image(path)
    resampled_mask = resample_mask(mask, header.spacing, resolution)
    header['spacing'] = resolution
    write_image(save_path, resampled_mask, header)
