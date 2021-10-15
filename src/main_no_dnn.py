from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from logging import Logger, getLogger

import lightgbm as lgb
import MeCab
import tensorflow_hub as hub
import tensorflow_text
import torch
from argparse_dataclass import ArgumentParser
from sklearn import dummy, ensemble, metrics, neighbors, svm, tree
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaModel, T5Tokenizer
from imblearn.ensemble import BalancedBaggingClassifier

from emotion_classification.dataset import TextClassificationDataset


class FeatureExtractorBase:
    def __init__(self, config):
        self.config = config


class FeatureExtractorTfidf(FeatureExtractorBase):
    def __init__(self, config):
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

    def vectorize(self, data):
        return self.vectorizer.fit_transform(data)


class FeatureExtractorUse(FeatureExtractorBase):
    def __init__(self, config):
        super().__init__(config)
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )

    def vectorize(self, data):
        return self.embed(data)


class FeatureExtractorRoberta(FeatureExtractorBase):
    def __init__(self, config):
        super().__init__(config)
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        model_name = "rinna/japanese-roberta-base"
        self.model = RobertaModel.from_pretrained(model_name, return_dict=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, padding=True)

        self.model.eval()

    @torch.no_grad()
    def vectorize(self, data):
        inputs = self.tokenizer(data, return_tensors="pt", padding=True).to(self.device)

        batch_size, token_size = inputs["input_ids"].shape
        position_ids = (
            torch.arange(token_size).expand((batch_size, -1)).to(self.device)
        )

        outputs = self.model(**inputs, position_ids=position_ids)

        return outputs.pooler_output


class Trainer:
    def __init__(self, config):
        self.config = config

        self.dataset_train = TextClassificationDataset(config, "train")
        self.dataset_eval = TextClassificationDataset(config, "eval")

        self.vectorizer = self.__create_vectorizer(config.vectorizer_type)

    def __create_vectorizer(self, vectorizer_type: str) -> FeatureExtractorBase:
        if args.vectorizer_type == "tfidf":
            return FeatureExtractorTfidf(self.config)
        if args.vectorizer_type == "use":
            return FeatureExtractorUse(self.config)
        if args.vectorizer_type == "roberta":
            return FeatureExtractorRoberta(self.config)

    def __create_models(self, model_type: list[str]):
        n_ens = 100
        return [
            (dummy.DummyClassifier(strategy="stratified"), "dummy"),
            (tree.DecisionTreeClassifier(), "decision tree"),
            (ensemble.RandomForestClassifier(n_estimators=n_ens), "random forest"),
            (ensemble.ExtraTreesClassifier(n_estimators=n_ens), "extra tree"),
            (
                lgb.LGBMClassifier(
                    objective="multiclass", num_class=self.dataset_train.n_labels
                ),
                "lightgbm",
            ),
            (svm.SVC(), "svc"),
            (neighbors.KNeighborsClassifier(), "knn"),
        ]

    def __run_model(self, model, name, X_train, X_eval, y_train, y_eval):
        start = time.time()
        model.fit(X_train, y_train)
        print(f"[{name}] {time.time() - start}")
        y_pred = model.predict(X_eval)
        print(metrics.classification_report(y_eval, y_pred))

    def train(self):
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


@dataclass
class Config:
    vectorizer_type: str = field(
        default="use", metadata=dict(type=str, choices=["tfidf", "use", "roberta"])
    )
    balanced: bool = field(default=False, metadata=dict(type=bool))
    dataroot: str = field(default="data/debug", metadata=dict(type=str))
    data_file: str = field(default=None, metadata=dict(type=str))
    label_file: str = field(default=None, metadata=dict(type=str))
    logger: Logger = field(default=getLogger(__name__), metadata=dict(type=Logger))


if __name__ == "__main__":
    parser = ArgumentParser(Config)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
    sys.exit()
