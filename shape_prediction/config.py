import os

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
xvertseg_dir = os.path.join(base_dir, 'datasets/xvertseg/')
verse2019_dir = os.path.join(base_dir, 'datasets/verse2019/')
nlst_dir = os.path.join(base_dir, 'datasets/nlst/')

# dir to store experiment
run_name = 'unet_data_augmentation'
experiments_dir = os.path.join(base_dir, 'experiments/shape_prediction/')
run_dir = os.path.join(experiments_dir, run_name)

# input config
patch_size = (188, 188, 188)

# wandb
wandb_key = '272782fa3a98a5f215cc2e580ebb4628245ea8e8'