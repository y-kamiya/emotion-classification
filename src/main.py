import sys

import mlflow
import pandas as pd
from argparse_dataclass import ArgumentParser

from emotion_classification.trainer import Trainer
from emotion_classification.config import Config


def main():
    parser = ArgumentParser(Config)
    args = parser.parse_args()

    pd.options.display.precision = 3
    pd.options.display.max_columns = 30

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
