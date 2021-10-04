import uuid
import logging
import torch
from dataclasses import dataclass, field
from logzero import setup_logger

from emotion_classification.dataset import (
    BaseDataset,
    EmotionDataset,
    SemEval2018EmotionDataset,
    TextClassificationDataset,
)


@dataclass
class Config:
    cpu: bool = False
    loglevel: str = field(
        default="INFO",
        metadata=dict(choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]),
    )
    log_interval: int = 1
    eval_interval: int = 1
    dataroot: str = "data"
    data_file: str = None
    label_file: str = None
    batch_size: int = 64
    epochs: int = 10
    fp16: bool = False
    lang: str = field(default="ja", metadata=dict(choices=["en", "ja"]))
    eval_only: bool = False
    predict: bool = False
    no_save: bool = False
    name: str = None
    freeze_base: bool = False
    lr: float = 1e-5
    multi_labels: bool = False
    dataset_class_name: str = field(
        default="EmotionDataset",
        metadata=dict(
            choices=[
                "EmotionDataset",
                "SemEval2018EmotionDataset",
                "TextClassificationDataset",
            ]
        ),
    )
    model_type: str = field(
        default="roberta", metadata=dict(choices=["bert", "roberta"])
    )
    custom_head: bool = False
    freeze_base_model: bool = False

    logger: logging.Logger = logging.getLogger(__name__)
    device_name: str = None
    device: str = None
    tensorboard_log_dir: str = None
    model_path: str = None
    best_model_path: str = None
    dataset_class: BaseDataset = None

    def __post_init__(self):
        if self.device_name is None:
            is_cpu = self.cpu or not torch.cuda.is_available()
            self.device_name = "cpu" if is_cpu else "cuda:0"

        if self.device is None:
            self.device = torch.device(self.device_name)

        if self.name is None:
            self.name = str(uuid.uuid4())[:8]

        if self.tensorboard_log_dir is None:
            self.tensorboard_log_dir = f"{self.dataroot}/runs/{self.name}"

        if self.model_path is None:
            self.model_path = f"{self.dataroot}/{self.name}.pth"

        if self.best_model_path is None:
            self.best_model_path = f"{self.dataroot}/{self.name}.best.pth"

        self.dataset_class = globals()[self.dataset_class_name]
        if self.dataset_class_name == "SemEval2018EmotionDataset":
            self.multi_labels = True

        if self.logger is None:
            logger = setup_logger(name=__name__, level=self.loglevel)
            self.logger = logger

        self.logger.info(self)
