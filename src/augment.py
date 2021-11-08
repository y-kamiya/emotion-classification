import os
from dataclasses import dataclass

import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from eda_ja.eda import EdaJa


@dataclass
class PreprocessConfig:
    input: str = "data/debug/train.tsv"


cs = ConfigStore.instance()
cs.store(name="base_config", node=PreprocessConfig)


class Augmenter:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        # self.eda = EdaJa()

    def build(self):
        df = pd.read_csv(self.config.input, sep="\t", names=["text", "label"])
        print(pd.value_counts(df["label"]))


@hydra.main(config_path=None, config_name="base_config")
def main(config: PreprocessConfig):
    if not os.path.isabs(config.input):
        config.input = os.path.join(hydra.utils.get_original_cwd(), config.input)
    OmegaConf.to_yaml(config)

    augmenter = Augmenter(config)
    augmenter.build()


if __name__ == "__main__":
    main()
