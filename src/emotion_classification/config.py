import uuid
import torch
from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    cpu: bool = False
    loglevel: str = field(
        default="INFO",
        metadata=dict(choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]),
    )
    log_interval: int = 1
    eval_interval: int = 1
    dataroot: str = "data"
    label_file: str = "labels.txt"
    batch_size: int = 64
    epochs: int = 1
    fp16: bool = False
    lang: str = field(default="ja", metadata=dict(choices=["en", "ja"]))
    eval_only: bool = False
    predict: bool = False
    no_save: bool = False
    name: str = str(uuid.uuid4())[:8]
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

    device: str = "cpu" if "${cpu}" or not torch.cuda.is_available() else "cuda"
    tensorboard_log_dir: str = "${dataroot}/runs/${name}"
    model_path: str = "${dataroot}/${name}.pth"
    best_model_path: str = "${dataroot}/${name}.best.pth"
