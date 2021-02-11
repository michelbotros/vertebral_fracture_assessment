import os

# pre processing config
patch_size = (128, 128, 128)
resolution = (1.0, 1.0, 1.0)

# training configs
train_val_split = 0.8
batch_size = 8
lr = 0.01
epochs = 250
use_weights = False
wandb_key = '272782fa3a98a5f215cc2e580ebb4628245ea8e8'

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
data_dir = os.path.join(base_dir, 'datasets/xvertseg/')


