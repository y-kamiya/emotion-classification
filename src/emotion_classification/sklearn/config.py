from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List

from emotion_classification.config import TrainerConfig


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
class SklearnConfig:
    predict: str = ""
    save_model: bool = False
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
