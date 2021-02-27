import os

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
xvertseg_dir = os.path.join(base_dir, 'datasets/xvertseg/')
verse2019_dir = os.path.join(base_dir, 'datasets/verse2019/')

# dir to store experiment
run_name = 'standard_cnn'
experiments_dir = os.path.join(base_dir, 'experiments/')

# loading config
patch_size = (128, 128, 128)
resolution = (1.0, 1.0, 1.0)

# training config
batch_size = 16
lr = 1e-3
epochs = 100
n_linear = 1024
init_filters = 32
groups = 1
batch_norm = True
data_aug = True
dropout = False

# wandb
wandb_key = '272782fa3a98a5f215cc2e580ebb4628245ea8e8'


