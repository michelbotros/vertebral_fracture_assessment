import os

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
xvertseg_dir = os.path.join(base_dir, 'datasets/xvertseg/')
verse2019_dir = os.path.join(base_dir, 'datasets/verse2019/')

# dir to store experiment
run_name = 'ResNet10_grades_NO_WS'
experiments_dir = os.path.join(base_dir, 'experiments/')

# loading config
patch_size = (128, 128, 128)
resolution = (1.0, 1.0, 1.0)

# training config
batch_size = 4
lr = 1e-4
epochs = 75
weight_decay = 0.001
data_aug = True

# wandb
wandb_key = '272782fa3a98a5f215cc2e580ebb4628245ea8e8'


