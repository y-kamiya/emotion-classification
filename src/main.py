from __future__ import annotations

import os
import sys

import hydra
import mlflow
import pandas as pd
from hydra.core.config_store import ConfigStore
from logzero import setup_logger
from omegaconf import OmegaConf

from emotion_classification.config import TrainerConfig
from emotion_classification.trainer import Trainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)


@hydra.main(config_path="conf", config_name="main")
def main(config: TrainerConfig):
    dataroot = config.dataroot
    if not os.path.isabs(dataroot):
        config.dataroot = os.path.join(hydra.utils.get_original_cwd(), dataroot)

    print(OmegaConf.to_yaml(config))

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

    mlflow.set_tracking_uri(f"{config.dataroot}/mlruns")
    mlflow.start_run(run_name=config.name)
    mlflow.log_params(OmegaConf.to_container(config))

    logger = setup_logger(name=__name__, level=config.loglevel.name)

    trainer = Trainer(config, logger)

    if config.eval_only:
        trainer.eval(0)
        sys.exit()

    if config.predict:
        trainer.predict()
        sys.exit()

    trainer.main()


if __name__ == "__main__":
    main()
