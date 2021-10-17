import sys

import mlflow
import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from logzero import setup_logger

from emotion_classification.trainer import Trainer
from emotion_classification.config import TrainerConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)


@hydra.main(config_path="conf", config_name="main")
def main(config: TrainerConfig):
    print(OmegaConf.to_yaml(config))

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

    mlflow.set_tracking_uri(f"{config.dataroot}/mlruns")
    mlflow.log_params(OmegaConf.to_container(config))

    logger = setup_logger(name=__name__, level=config.loglevel.name)

    trainer = Trainer(config, logger)

    if config.eval_only:
        trainer.eval(0)
        sys.exit()

    if config.predict:
        trainer.predict()
        sys.exit()

    for epoch in range(config.epochs):
        trainer.train(epoch)
        if epoch % config.eval_interval == 0:
            trainer.eval(epoch)

    trainer.eval(epoch)


if __name__ == "__main__":
    main()
