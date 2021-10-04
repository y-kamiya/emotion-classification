import argparse
import sys
import uuid

import mlflow
import pandas as pd
import torch
from logzero import setup_logger

from emotion_classification.dataset import (
    EmotionDataset,
    SemEval2018EmotionDataset,
    TextClassificationDataset,
)
from emotion_classification.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--loglevel", default="DEBUG")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument(
        "--n_labels", type=int, default=5, help="number of classes to train"
    )
    parser.add_argument("--dataroot", default="data", help="path to data directory")
    parser.add_argument("--data_file", default=None, help="file name with data")
    parser.add_argument("--label_file", default=None, help="file name with labels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of batch")
    parser.add_argument("--epochs", type=int, default=10, help="epoch count")
    parser.add_argument("--fp16", action="store_true", help="run model with float16")
    parser.add_argument("--lang", default="ja", choices=["en", "ja"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--multi_labels", action="store_true")
    parser.add_argument(
        "--dataset_class_name",
        default="EmotionDataset",
        choices=[
            "EmotionDataset",
            "SemEval2018EmotionDataset",
            "TextClassificationDataset",
        ],
    )
    parser.add_argument("--model_type", default="roberta", choices=["bert", "roberta"])
    parser.add_argument("--custom_head", action="store_true")
    parser.add_argument("--freeze_base_model", action="store_true")
    args = parser.parse_args()

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda:0"
    args.device = torch.device(args.device_name)

    logger = setup_logger(name=__name__, level=args.loglevel)
    args.logger = logger

    if args.name is None:
        args.name = str(uuid.uuid4())[:8]

    args.tensorboard_log_dir = f"{args.dataroot}/runs/{args.name}"

    args.model_path = f"{args.dataroot}/{args.name}.pth"
    args.best_model_path = f"{args.dataroot}/{args.name}.best.pth"

    args.dataset_class = globals()[args.dataset_class_name]
    if args.dataset_class_name == "SemEval2018EmotionDataset":
        args.multi_labels = True

    logger.info(args)
    mlflow.set_tracking_uri(f"{args.dataroot}/mlruns")
    mlflow.log_params(vars(args))

    trainer = Trainer(args)

    if args.eval_only:
        trainer.eval(0)
        sys.exit()

    if args.predict:
        trainer.predict()
        sys.exit()

    for epoch in range(args.epochs):
        trainer.train(epoch)
        if epoch % args.eval_interval == 0:
            trainer.eval(epoch)

    trainer.eval(epoch)


if __name__ == "__main__":
    main()
