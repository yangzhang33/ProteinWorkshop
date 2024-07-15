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
        # "decoder=dummy_decoder",
        "decoder.graph_label.dummy=True",
        "task=multiclass_graph_classification",
        "dataset=fold_family",
        "features=ca_base", 
        "+aux_task=none",
        "trainer.max_epochs=1000",
        "encoder.num_blocks=4",
        "encoder.dropout=0.3",
        "encoder.out_channels=1195",
        "optimiser=adam",
        "optimiser.optimizer.lr=5e-4",
        "callbacks.early_stopping.patience=10",
        "test=True",
        "scheduler=steplr",
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