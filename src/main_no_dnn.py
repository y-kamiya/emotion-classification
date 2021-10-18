from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import hydra
import lightgbm as lgb
import MeCab
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import torch
from hydra.core.config_store import ConfigStore
from imblearn.ensemble import BalancedBaggingClassifier
from logzero import setup_logger
from omegaconf import OmegaConf
from sklearn import dummy, ensemble, metrics, neighbors, svm, tree
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaModel, T5Tokenizer

from emotion_classification.config import TrainerConfig
from emotion_classification.dataset import TextClassificationDataset

logger = setup_logger(__name__)


class FeatureExtractorBase(ABC):
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def vectorize(self, data: list[str]) -> np.array:
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

    def vectorize(self, data: list[str]) -> np.array:
        return self.vectorizer.fit_transform(data)


class FeatureExtractorUse(FeatureExtractorBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        os.environ["TFHUB_CACHE_DIR"] = "/tmp/tf_cache"
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )

    def vectorize(self, data: list[str]) -> np.array:
        return self.embed(data)


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
    def vectorize(self, data: list[str]) -> np.array:
        inputs = self.tokenizer(data, return_tensors="pt", padding=True).to(self.device)

        batch_size, token_size = inputs["input_ids"].shape
        position_ids = torch.arange(token_size).expand((batch_size, -1)).to(self.device)

        outputs = self.model(**inputs, position_ids=position_ids)

        return outputs.pooler_output


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.dataset_train = TextClassificationDataset(config.trainer, "train", logger)
        self.dataset_eval = TextClassificationDataset(config.trainer, "eval", logger)

        self.vectorizer = self.__create_vectorizer(config.vectorizer_type)

    def __create_vectorizer(
        self, vectorizer_type: VectorizerType
    ) -> FeatureExtractorBase:
        type = self.config.vectorizer_type

        if type == VectorizerType.TFIDF:
            return FeatureExtractorTfidf(self.config)

        if type == VectorizerType.ROBERTA:
            return FeatureExtractorRoberta(self.config)

        return FeatureExtractorUse(self.config)

    def __create_models(self, model_type: list[str]) -> list[tuple[Any, str]]:
        n_ens = 100
        model_type = self.config.model_type
        models = []

        if model_type in [ModelType.ALL, ModelType.DUMMY]:
            models.append((dummy.DummyClassifier(strategy="stratified"), "dummy"))

        if model_type in [ModelType.ALL, ModelType.RANDOM_FOREST]:
            models.append((ensemble.RandomForestClassifier(n_estimators=n_ens), "random forest"))

        if model_type in [ModelType.ALL, ModelType.EXTRA_TREES]:
            models.append((ensemble.ExtraTreesClassifier(n_estimators=n_ens), "extra tree"))

        if model_type in [ModelType.ALL, ModelType.LGBM]:
            n_labels = self.dataset_train.n_labels
            models.append((lgb.LGBMClassifier(objective="multiclass", num_class=n_labels), "lightgbm"))

        if model_type in [ModelType.ALL, ModelType.SVM]:
            models.append((svm.SVC(), "svc"))

        if model_type in [ModelType.ALL, ModelType.KNN]:
            models.append((neighbors.KNeighborsClassifier(), "knn"))

        return models

    def __run_model(
        self,
        model: Any,
        name: str,
        X_train: np.array,
        X_eval: np.array,
        y_train: np.array,
        y_eval: np.array,
    ) -> None:
        start = time.time()
        model.fit(X_train, y_train)
        print(f"[{name}] {time.time() - start}")
        y_pred = model.predict(X_eval)
        print(metrics.classification_report(y_eval, y_pred))

    def train(self) -> None:
        vectors_train = self.vectorizer.vectorize(self.dataset_train.texts)
        vectors_eval = self.vectorizer.vectorize(self.dataset_eval.texts)

        for (model, name) in self.__create_models([]):
            if self.config.balanced:
                model = BalancedBaggingClassifier(base_estimator=model)

            self.__run_model(
                model,
                name,
                vectors_train,
                vectors_eval,
                self.dataset_train.labels.argmax(axis=1),
                self.dataset_eval.labels.argmax(axis=1),
            )


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


@dataclass
class Config:
    vectorizer_type: VectorizerType = VectorizerType.USE
    model_type: ModelType = ModelType.ALL
    balanced: bool = False
    trainer: TrainerConfig = TrainerConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="main_no_dnn")
def main(config: Config):
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)
    trainer.train()
    sys.exit()


if __name__ == "__main__":
    main()
