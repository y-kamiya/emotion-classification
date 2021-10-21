from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List

import hydra
import imblearn
import lightgbm as lgb
import MeCab
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
import torch
from hydra.core.config_store import ConfigStore
from logzero import setup_logger
from omegaconf import OmegaConf
from scipy.stats import uniform
from sklearn import dummy, ensemble, metrics, neighbors, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tabulate import tabulate
from transformers import RobertaModel, T5Tokenizer

from emotion_classification.config import TrainerConfig
from emotion_classification.dataset import BaseDataset, Phase, TextClassificationDataset

logger = setup_logger(__name__)


class FeatureExtractorBase(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def vectorize(self, dataset: BaseDataset) -> np.array:
        pass


class FeatureExtractorTfidf(FeatureExtractorBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tagger = MeCab.Tagger()
        self.vectorizer = TfidfVectorizer(
            use_idf=True, min_df=0.02, stop_words=[], token_pattern="(?u)\\b\\w+\\b"
        )

    def parse(self, text: str) -> str:
        node = self.tagger.parseToNode(text)

        words = []
        while node:
            pos = node.feature.split(",")
            if pos[0] == "動詞":
                words.append(pos[6])
            elif pos[0] != "助詞":
                words.append(node.surface.lower())
            node = node.next

        return " ".join(words)

    def vectorize(self, dataset: BaseDataset) -> np.array:
        if dataset.phase == Phase.TRAIN:
            return self.vectorizer.fit_transform(dataset.texts)

        return self.vectorizer.transform(dataset.texts)


class FeatureExtractorUse(FeatureExtractorBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        os.environ["TFHUB_CACHE_DIR"] = "/tmp/tf_cache"
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )

    def vectorize(self, dataset: BaseDataset) -> np.array:
        return self.embed(dataset.texts).numpy()


class FeatureExtractorRoberta(FeatureExtractorBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        model_name = "rinna/japanese-roberta-base"
        self.model = RobertaModel.from_pretrained(model_name, return_dict=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, padding=True)

        self.model.eval()

    @torch.no_grad()
    def vectorize(self, dataset: BaseDataset) -> np.array:
        inputs = self.tokenizer(dataset.texts, return_tensors="pt", padding=True).to(
            self.device
        )

        batch_size, token_size = inputs["input_ids"].shape
        position_ids = torch.arange(token_size).expand((batch_size, -1)).to(self.device)

        outputs = self.model(**inputs, position_ids=position_ids)

        return outputs.pooler_output.numpy()


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.dataset_train = TextClassificationDataset(
            config.trainer, Phase.TRAIN, logger
        )
        self.dataset_eval = TextClassificationDataset(
            config.trainer, Phase.EVAL, logger
        )

        self.vectorizer = self.__create_vectorizer(config.vectorizer_type)

        if config.sampling:
            over_sampler = imblearn.over_sampling.SMOTE(
                random_state=0,
                sampling_strategy=self.__strategy(config.over_sampling_strategy),
            )
            under_sampler = imblearn.under_sampling.RandomUnderSampler(
                random_state=0,
                sampling_strategy=self.__strategy(config.under_sampling_strategy),
            )

            def sampler_func(X, y):
                logger.info(f"original: {self.__label_counts(y)}")
                X, y = over_sampler.fit_resample(X, y)
                logger.info(f"oversampled: {self.__label_counts(y)}")
                X, y = under_sampler.fit_resample(X, y)
                logger.info(f"undersampled: {self.__label_counts(y)}")
                return X, y

            self.sampler = imblearn.FunctionSampler(func=sampler_func)

    def __strategy(self, strategy):
        if not strategy:
            return "auto"

        return OmegaConf.to_container(strategy)

    def __create_vectorizer(
        self, vectorizer_type: VectorizerType
    ) -> FeatureExtractorBase:
        type = self.config.vectorizer_type

        if type == VectorizerType.TFIDF:
            return FeatureExtractorTfidf(self.config)

        if type == VectorizerType.ROBERTA:
            return FeatureExtractorRoberta(self.config)

        return FeatureExtractorUse(self.config)

    def __create_models(self, model_type: list[str]) -> list[tuple[Any, ModelType]]:
        n_jobs = self.config.n_jobs
        model_type = self.config.model_type
        models = []

        if model_type in [ModelType.ALL, ModelType.DUMMY]:
            models.append(
                (dummy.DummyClassifier(strategy="stratified"), ModelType.DUMMY)
            )

        if model_type in [ModelType.ALL, ModelType.RANDOM_FOREST]:
            models.append(
                (
                    ensemble.RandomForestClassifier(n_jobs=n_jobs),
                    ModelType.RANDOM_FOREST,
                )
            )

        if model_type in [ModelType.ALL, ModelType.EXTRA_TREES]:
            models.append(
                (
                    ensemble.ExtraTreesClassifier(n_jobs=n_jobs),
                    ModelType.EXTRA_TREES,
                )
            )

        if model_type in [ModelType.ALL, ModelType.LGBM]:
            n_labels = self.dataset_train.n_labels
            models.append(
                (
                    lgb.LGBMClassifier(
                        objective="multiclass", num_class=n_labels, n_jobs=n_jobs
                    ),
                    ModelType.LGBM,
                )
            )

        if model_type in [ModelType.ALL, ModelType.SVM]:
            models.append((svm.SVC(), ModelType.SVM))

        if model_type in [ModelType.ALL, ModelType.KNN]:
            models.append(
                (neighbors.KNeighborsClassifier(n_jobs=n_jobs), ModelType.KNN)
            )

        return models

    def __create_params_grid_search(self, model_type) -> dict[str, Any]:
        if model_type == ModelType.SVM:
            return {
                "C": [1e-2, 1e-1, 1, 10, 100],
                "gamma": [1e-2, 1e-1, 1, 10, 100],
            }

        if model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": [100, 500, 1000, 5000],
            }

        if model_type == ModelType.LGBM:
            return {
                "learning_rate": [1e-2, 1e-1, 1, 10, 100],
            }

        assert False, f"params for {model_type} is not defined"

    def __create_params_random_search(self, model_type) -> dict[str, Any]:
        if model_type == ModelType.SVM:
            return {
                "C": uniform(1e-3, 1e3),
                "gamma": uniform(1e-3, 1e3),
            }

        assert False, f"params for {model_type} is not defined"

    def __label_counts(self, data):
        return str(sorted(Counter(data).items()))

    def __run_model(
        self,
        model: Any,
        model_type: ModelType,
        X_train: np.array,
        X_eval: np.array,
        y_train: np.array,
        y_eval: np.array,
    ) -> None:
        start = time.time()
        model.fit(X_train, y_train)
        print(f"[{model_type.name}] {time.time() - start}")
        y_pred = model.predict(X_eval)
        print(metrics.classification_report(y_eval, y_pred))

    def train(self) -> None:
        X_train = self.vectorizer.vectorize(self.dataset_train)
        X_eval = self.vectorizer.vectorize(self.dataset_eval)
        y_train = self.dataset_train.labels.argmax(axis=1).numpy()
        y_eval = self.dataset_eval.labels.argmax(axis=1).numpy()

        if self.config.sampling:
            X_train, y_train = self.sampler.fit_resample(X_train, y_train)

        for (model, model_type) in self.__create_models([]):
            if self.config.bagging:
                model = ensemble.BaggingClassifier(base_estimator=model)

            scoring = OmegaConf.to_container(self.config.search_scoring)
            n_jobs = self.config.n_jobs
            if self.config.search_type == SearchType.GRID:
                params = self.__create_params_grid_search(model_type)
                model = GridSearchCV(
                    model, params, n_jobs=n_jobs, scoring=scoring, refit=scoring[0]
                )
            elif self.config.search_type == SearchType.RANDOM:
                params = self.__create_params_random_search(model_type)
                model = RandomizedSearchCV(
                    model, params, n_jobs=n_jobs, scoring=scoring, refit=scoring[0]
                )

            self.__run_model(
                model,
                model_type,
                X_train,
                X_eval,
                y_train,
                y_eval,
            )

            if self.config.search_type != SearchType.NONE:
                keys = sum(
                    [[f"rank_test_{name}", f"mean_test_{name}"] for name in scoring], []
                )
                df = pd.DataFrame(model.cv_results_)
                df.sort_values(by=keys[0], inplace=True)
                df = df[keys + ["params"]]
                df.to_csv("search_output", sep="\t")
                print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))


class VectorizerType(Enum):
    TFIDF = auto()
    USE = auto()
    ROBERTA = auto()


class ModelType(Enum):
    ALL = auto()
    DUMMY = auto()
    RANDOM_FOREST = auto()
    EXTRA_TREES = auto()
    LGBM = auto()
    SVM = auto()
    KNN = auto()


class SearchType(Enum):
    NONE = auto()
    GRID = auto()
    RANDOM = auto()


@dataclass
class Config:
    vectorizer_type: VectorizerType = VectorizerType.USE
    model_type: ModelType = ModelType.KNN
    search_type: SearchType = SearchType.NONE
    search_scoring: List[str] = field(default_factory=lambda: ["f1_micro", "f1_macro"])
    bagging: bool = False
    sampling: bool = False
    over_sampling_strategy: Dict[int, int] = field(default_factory=lambda: {})
    under_sampling_strategy: Dict[int, int] = field(default_factory=lambda: {})
    n_jobs: int = 4
    trainer: TrainerConfig = TrainerConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="main_no_dnn")
def main(config: Config):
    dataroot = config.trainer.dataroot
    if not os.path.isabs(dataroot):
        config.trainer.dataroot = os.path.join(hydra.utils.get_original_cwd(), dataroot)

    print(OmegaConf.to_yaml(config))

    if config.search_type != SearchType.NONE:
        assert (
            config.model_type != ModelType.ALL
        ), "model_type should not be ALL when search_type is not NONE"

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

    trainer = Trainer(config)
    trainer.train()
    sys.exit()


if __name__ == "__main__":
    main()
