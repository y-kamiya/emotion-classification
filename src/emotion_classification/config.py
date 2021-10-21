import uuid
from dataclasses import dataclass
from enum import Enum, auto

import torch


class DatasetType(Enum):
    EMOTION = auto()
    SEM_EVAL_2018_EMOTION = auto()
    TEXT_CLASSIFICATION = auto()


class ModelType(Enum):
    BERT = auto()
    ROBERTA = auto()


class Lang(Enum):
    JA = auto()
    EN = auto()


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class TrainerConfig:
    cpu: bool = False
    loglevel: LogLevel = LogLevel.INFO
    log_interval: int = 1
    eval_interval: int = 1
    dataroot: str = "data"
    label_file: str = "labels.txt"
    batch_size: int = 64
    epochs: int = 1
    fp16: bool = False
    lang: Lang = Lang.JA
    eval_only: bool = False
    predict: bool = False
    no_save: bool = False
    name: str = str(uuid.uuid4())[:8]
    freeze_base: bool = False
    lr: float = 1e-5
    multi_labels: bool = False
    dataset_type: DatasetType = DatasetType.EMOTION
    model_type: ModelType = ModelType.ROBERTA
    custom_head: bool = False
    freeze_base_model: bool = False
    sampler_alpha: float = 0

    device: str = "cpu" if "${cpu}" or not torch.cuda.is_available() else "cuda"
    tensorboard_log_dir: str = "${dataroot}/runs/${name}"
    model_path: str = "${dataroot}/${name}.pth"
    best_model_path: str = "${dataroot}/${name}.best.pth"
