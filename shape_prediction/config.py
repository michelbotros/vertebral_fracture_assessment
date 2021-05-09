import os

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
xvertseg_dir = os.path.join(base_dir, 'datasets/xvertseg/')
verse2019_dir = os.path.join(base_dir, 'datasets/verse2019/')
nlst_dir = os.path.join(base_dir, 'datasets/nlst/')

# input config
img_shape = (32, 128, 128)