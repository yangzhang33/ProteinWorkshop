# Misc. tools
import os

# Hydra tools
import hydra

from hydra.compose import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from proteinworkshop.constants import HYDRA_CONFIG_PATH
from proteinworkshop.utils.notebook import init_hydra_singleton

version_base = "1.2"  # Note: Need to update whenever Hydra is upgraded
init_hydra_singleton(reload=True, version_base=version_base)

path = HYDRA_CONFIG_PATH
rel_path = os.path.relpath(path, start=".")
# print(rel_path)
GlobalHydra.instance().clear()
hydra.initialize(rel_path, version_base=version_base)

cfg = hydra.compose(
    config_name="train",
    overrides=[
        "encoder=schnet",
        "encoder.hidden_channels=512", # Number of channels in the hidden layers
        "encoder.out_dim=32", # Output dimension of the model
        "encoder.num_layers=6", # Number of filters used in convolutional layers
        "encoder.num_filters=128", # Number of convolutional layers in the model
        "encoder.num_gaussians=50", # Number of Gaussian functions used for radial filters
        "encoder.cutoff=10.0", # Cutoff distance for interactions
        "encoder.max_num_neighbors=32", # Maximum number of neighboring atoms to consider
        "encoder.readout=add", # Global pooling method to be used
        "encoder.dipole=False",
        "encoder.mean=null",
        "encoder.std=null",
        "encoder.atomref=null",
        "encoder.pretraining=False",

        "task=multiclass_graph_classification",
        "dataset=ec_reaction",
        "dataset.datamodule.batch_size=32",
        "features=ca_base", 
        "+aux_task=none",
        "+ckpt_path=/home/zhang/Projects/3d/ProteinWorkshop/notebooks/outputs/checkpoints/last.ckpt",
        "trainer.max_epochs=400",
        "optimiser=adam",
        "optimiser.optimizer.lr=5e-4",
        "callbacks.early_stopping.patience=200",
        "test=True",
        "scheduler=steplr",

        ## for test ONLY
        # "task_name=test",  # here
        # "ckpt_path_test=/home/zhang/Projects/3d/proteinworkshop_checkpoints/outputs_pronet_fold_400epochs/checkpoints/epoch_273.ckpt", # here
        # "optimizer.weight_decay=0.5"
    ],
    return_hydra_config=True,
)

# Note: Customize as needed e.g., when running a sweep
cfg.hydra.job.num = 0
cfg.hydra.job.id = 0
cfg.hydra.hydra_help.hydra_help = False
cfg.hydra.runtime.output_dir = "outputs"

HydraConfig.instance().set_config(cfg)

from proteinworkshop.configs import config

cfg = config.validate_config(cfg)

print(cfg.keys())
for key in cfg.keys():
    print(key)
    print(cfg[key])

from proteinworkshop.train import train_model

train_model(cfg)