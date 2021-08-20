import os

# data config
base_dir = '/mnt/netcache/bodyct/experiments/vertebra_fracture_detection_t9560/'
xvertseg_dir = os.path.join(base_dir, 'datasets/xvertseg/')
verse2019_dir = os.path.join(base_dir, 'datasets/verse2019/')
nlst_dir = os.path.join(base_dir, 'datasets/nlst/')

# dir to store this experiment
experiments_dir = os.path.join(base_dir, 'experiments/shape_prediction/')
run_name = 'refine_final'
run_dir = os.path.join(experiments_dir, run_name)

# for loading the coarse model
coarse_run_dir = os.path.join(experiments_dir, 'coarse_final')
coarse_model_path = os.path.join(coarse_run_dir, 'best_model_epoch_41_loss_0.032.pt')

# for loading refinement model
refine_run_dir = os.path.join(experiments_dir, 'refine_final')
refine_model_path = os.path.join(refine_run_dir, 'best_model_epoch_191_loss_0.033.pt')

# input config
patch_size = (64, 64, 64)
resolution = (1.0, 1.0, 1.0)
context = 2

# wandb
wandb_key = '272782fa3a98a5f215cc2e580ebb4628245ea8e8'