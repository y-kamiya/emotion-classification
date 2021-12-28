from __future__ import annotations

import os
import sys

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from logzero import setup_logger
from omegaconf import OmegaConf

from emotion_classification.sklearn.config import ModelType, SearchType, SklearnConfig
from emotion_classification.sklearn.trainer import SklearnTrainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=SklearnConfig)


@hydra.main(config_path="conf", config_name="main_sklearn")
def main(config: SklearnConfig):
    dataroot = config.trainer.dataroot
    if not os.path.isabs(dataroot):
        config.trainer.dataroot = os.path.join(hydra.utils.get_original_cwd(), dataroot)

    model_path = config.trainer.model_path
    if not os.path.isabs(model_path):
        config.trainer.model_path = os.path.join(
            hydra.utils.get_original_cwd(), model_path
        )

    print(OmegaConf.to_yaml(config))

    if config.search_type != SearchType.NONE:
        assert (
            config.model_type != ModelType.ALL
        ), "model_type should not be ALL when search_type is not NONE"

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

    logger = setup_logger(__name__)

    trainer = SklearnTrainer(config, logger)

    if config.trainer.predict:
        print(trainer.predict())
        sys.exit()

    trainer.train()


if __name__ == "__main__":
    main()
