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
        "encoder=pronet",
        "encoder.level='aminoacid'",
        "encoder.num_blocks=4",
        "encoder.hidden_channels=128",
        "encoder.out_channels=1195",
        "encoder.mid_emb=64",
        "encoder.num_radial=6",
        "encoder.num_spherical=2",
        "encoder.cutoff=10.0",
        "encoder.max_num_neighbors=32",
        "encoder.int_emb_layers=3",
        "encoder.out_layers=2",
        "encoder.num_pos_emb=16",
        "encoder.dropout=0.3",
        "encoder.data_augment_eachlayer=True",
        "encoder.euler_noise=False",
        "encoder.pretraining=False",
        "encoder.node_embedding=False",

        "decoder.graph_label.dummy=True",

        "task=multiclass_graph_classification",
        "dataset=fold_family",
        "dataset.datamodule.batch_size=32",
        "features=ca_base", 
        "+aux_task=none",
        
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

from proteinworkshop.finetune import train_model

train_model(cfg)