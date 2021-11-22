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


class OptimizerType(Enum):
    RADAM = auto()
    ADAMW = auto()


@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loglevel: LogLevel = LogLevel.INFO
    log_interval: int = 1
    eval_interval: int = 1
    dataroot: str = "data"
    label_file: str = "labels.txt"
    batch_size: int = 64
    epochs: int = 1
    warmup_steps: int = 1000
    optimizer_type: OptimizerType = OptimizerType.RADAM
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

    tensorboard_log_dir: str = "${dataroot}/runs/${name}"
    model_path: str = "${dataroot}/${name}.pth"
    best_model_path: str = "${dataroot}/${name}.best.pth"
