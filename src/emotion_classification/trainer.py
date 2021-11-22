from __future__ import annotations

import io
import os
import time
from logging import Logger, getLogger
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pycm
import torch
import torch.optim as optim
from sklearn import metrics
from tabulate import tabulate
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer, AdamW, get_linear_schedule_with_warmup

import apex

from .config import DatasetType, ModelType, TrainerConfig
from .dataset import (
    BaseDataset,
    EmotionDataset,
    Phase,
    SemEval2018EmotionDataset,
    TextClassificationDataset,
)
from .model import BertModel, RobertaModel
from .sampler import SamplerFactory


class Trainer:
    def __init__(self, config: TrainerConfig, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = getLogger(__name__) if logger is None else logger

        if self.config.predict:
            dataset = self.__create_dataset(Phase.PREDICT)
            self.dataloader_predict = DataLoader(
                dataset, batch_size=self.config.batch_size
            )
        else:
            dataset = self.__create_dataset(Phase.TRAIN)
            self.dataloader_train = self.__create_dataloader(dataset)

            data_eval = self.__create_dataset(Phase.EVAL)
            self.dataloader_eval = DataLoader(
                data_eval, batch_size=self.config.batch_size, shuffle=False
            )

        self.model, self.tokenizer = self.__create_model(dataset.n_labels)

        self.optimizer = self.__create_optimizer(self.model)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=int(len(dataset) / self.config.batch_size * self.config.epochs),
        )

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.best_f1_score = 0.0

        if config.fp16:
            self.model, self.optimizer = apex.amp.initialize(
                self.model, self.optimizer, "O1"
            )

        self.start_epoch = 0
        self.load(self.config.model_path)

        if self.config.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def __create_optimizer(self, model):
        opt_parameters = []
        named_parameters = list(model.named_parameters()) 
        
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
        set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
        init_lr = self.config.lr
        
        for i, (name, params) in enumerate(named_parameters):  
            
            weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
     
            if name.startswith("roberta.embeddings") or name.startswith("roberta.encoder"):            
                lr = init_lr       
                lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
                lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})  
                
            if name.startswith("classifier"):
                lr = init_lr * 3.6 
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})    

        return AdamW(opt_parameters, lr=init_lr)

    def __create_dataloader(self, dataset):
        alpha = max(0, min(self.config.sampler_alpha, 1))

        if alpha == 0:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        sampler = SamplerFactory(self.logger).get(
            class_idxs=dataset.index_list_by_label(),
            batch_size=self.config.batch_size,
            n_batches=int(len(dataset) / self.config.batch_size),
            alpha=alpha,
            kind="random",
        )

        return DataLoader(dataset, batch_sampler=sampler)

    def __create_dataset(self, phase: Phase) -> BaseDataset:
        if self.config.dataset_type == DatasetType.EMOTION:
            return EmotionDataset(self.config, phase, self.logger)

        if self.config.dataset_type == DatasetType.SEM_EVAL_2018_EMOTION:
            self.config.multi_labels = True
            return SemEval2018EmotionDataset(self.config, phase, self.logger)

        return TextClassificationDataset(self.config, phase, self.logger)

    def __create_model(
        self, n_labels: int
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        if self.config.model_type == ModelType.BERT:
            return BertModel.create(self.config, n_labels)

        return RobertaModel.create(self.config, n_labels)

    def forward(
        self, inputs: BatchEncoding, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = None
        if self.config.model_type == "roberta":
            batch_size, token_size = inputs["input_ids"].shape
            position_ids = (
                torch.arange(token_size).expand((batch_size, -1)).to(self.config.device)
            )

        if self.config.multi_labels:
            outputs = self.model(**inputs, position_ids=position_ids)
            loss = F.binary_cross_entropy_with_logits(outputs.logits, labels)
            return loss, 0 < outputs.logits

        outputs = self.model(
            **inputs, labels=torch.argmax(labels, dim=1), position_ids=position_ids
        )
        return outputs.loss, torch.argmax(outputs.logits, dim=1)

    def train(self, epoch: int) -> None:
        self.model.train()

        for i, (texts, labels) in enumerate(self.dataloader_train):
            start_time = time.time()
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.config.device
            )
            labels = labels.to(self.config.device)

            loss, _ = self.forward(inputs, labels)

            if self.config.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.scheduler.step()

            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    "train epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}, lr: {:.2e}".format(
                        epoch, i, loss, elapsed_time, self.scheduler.get_lr()[0]
                    )
                )

            self.writer.add_scalar("loss/train", loss, epoch, start_time)
            mlflow.log_metric("loss/train", loss.item(), epoch)

        self.save(self.config.model_path, epoch)

    @torch.no_grad()
    def eval(self, epoch: int) -> None:
        self.model.eval()

        all_labels = torch.empty(0)
        all_preds = torch.empty(0)
        losses = []
        start_time = time.time()

        for i, (texts, labels) in enumerate(self.dataloader_eval):
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.config.device
            )
            labels = labels.to(self.config.device)

            loss, preds = self.forward(inputs, labels)

            losses.append(loss.item())

            if not self.config.multi_labels:
                labels = torch.argmax(labels, dim=1)

            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

        elapsed_time = time.time() - start_time
        average_loss = sum(losses) / len(losses)
        self.logger.info(
            "eval epoch: {}, loss: {:.2f}, time: {:.2f}".format(
                epoch, average_loss, elapsed_time
            )
        )

        self.__log_confusion_matrix(all_preds, all_labels, epoch)

        df = pd.DataFrame(
            metrics.classification_report(all_labels, all_preds, output_dict=True)
        )
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))

        if not self.config.eval_only:
            f1_score = df.loc["f1-score"]
            micro = (
                f1_score["micro avg"]
                if "micro avg" in f1_score
                else f1_score["accuracy"]
            )
            macro = f1_score["macro avg"]
            mlflow.log_metric("loss/eval", average_loss, epoch)
            mlflow.log_metric("metrics/f1_score_micro", micro, epoch)
            mlflow.log_metric("metrics/f1_score_macro", macro, epoch)
            self.writer.add_scalar("loss/eval", average_loss, epoch, start_time)
            self.writer.add_scalar(
                "metrics/f1_score_micro(accuracy)", micro, epoch, start_time
            )
            self.writer.add_scalar("metrics/f1_score_macro", macro, epoch, start_time)

            if self.best_f1_score < micro:
                self.best_f1_score = micro
                self.save(self.config.best_model_path, epoch)

    @torch.no_grad()
    def predict(self) -> list[str]:
        label_map = {
            value: key
            for key, value in self.dataloader_predict.dataset.label_index_map.items()
        }
        label_map[-1] = "none"
        np.set_printoptions(precision=0)

        pred_label_names = []
        output_path = os.path.join(self.config.dataroot, "predict_result")
        with open(output_path, "w") as f:
            for i, (texts, labels) in tqdm(enumerate(self.dataloader_predict)):
                if not self.config.multi_labels:
                    labels = torch.tensor(
                        [
                            -1 if sum(onehot) == 0 else torch.argmax(onehot)
                            for onehot in labels
                        ]
                    )
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                    self.config.device
                )
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                probs = F.softmax(outputs.logits, dim=1) * 100
                for j in range(len(texts)):
                    pred_label_name = label_map[preds[j].item()]
                    true_label_name = label_map[labels[j].item()]
                    prob = probs[j].cpu().numpy()
                    f.write(
                        f"{pred_label_name}\t{prob}\t{true_label_name}\t{texts[j]}\n"
                    )
                    pred_label_names.append(pred_label_name)

        self.logger.info(f"write predicted result to {output_path}")

        return pred_label_names

    def __log_confusion_matrix(
        self, all_preds: torch.Tensor, all_labels: torch.Tensor, epoch: int
    ):
        buf = io.BytesIO()
        dataset = self.dataloader_eval.dataset
        label_map = {value: key for key, value in dataset.label_index_map.items()}

        np.set_printoptions(precision=3)

        if self.config.multi_labels:
            fig, axes = plt.subplots(1, len(label_map.keys()), figsize=(25, 5))
            cm = metrics.multilabel_confusion_matrix(
                y_pred=all_preds.numpy(), y_true=all_labels.numpy()
            )
            for i in range(len(label_map.keys())):
                mat = np.array([[cm[i][1][1], cm[i][1][0]], [cm[i][0][1], cm[i][0][0]]])
                result = mat / mat.sum(axis=1, keepdims=True)
                print(f"{label_map[i]}\n{result}\n")

                display = metrics.ConfusionMatrixDisplay(
                    result, display_labels=["P", "N"]
                )
                display.plot(ax=axes[i], cmap=plt.cm.Blues, values_format=".2f")
                display.ax_.set_title(label_map[i])
                display.ax_.set_ylabel("True label" if i == 0 else "")
                display.ax_.set_yticklabels(["P", "N"] if i == 0 else [])
                display.im_.colorbar.remove()

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.colorbar(display.im_, ax=axes)
            plt.savefig(buf, format="png", dpi=180)
        else:
            cm = metrics.confusion_matrix(
                y_pred=all_preds.numpy(), y_true=all_labels.numpy(), normalize="true"
            )
            display = metrics.ConfusionMatrixDisplay(
                cm, display_labels=label_map.values()
            )
            display.plot(cmap=plt.cm.Blues)
            display.figure_.savefig(buf, format="png", dpi=180)

            cm = pycm.ConfusionMatrix(
                actual_vector=all_labels.numpy(), predict_vector=all_preds.numpy()
            )
            cm.relabel(mapping=label_map)
            cm.print_normalized_matrix()

        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite("confusion.png", img)
        mlflow.log_artifact("confusion.png")
        self.writer.add_image("confusion_maatrix", img, epoch, dataformats="HWC")

    def save(self, model_path: str, epoch: int) -> None:
        if self.config.no_save:
            return

        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "amp": apex.amp.state_dict() if self.config.fp16 else None,
            "batch_size": self.config.batch_size,
            "fp16": self.config.fp16,
            "last_epoch": epoch,
        }
        torch.save(data, model_path)
        self.logger.info(f"save model to {model_path}")

    def load(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            self.logger.warning(f"model_path: {model_path} is not found")
            return

        data = torch.load(model_path, map_location=self.config.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.start_epoch = data["last_epoch"] + 1
        if self.config.fp16:
            apex.amp.load_state_dict(data["amp"])

        self.logger.info(f"load model from {model_path}")

    def _can_eval(self, epoch):
        if epoch % self.config.eval_interval == 0:
            return True

        if epoch == self.config.epochs - 1:
            return True

        return False

    def main(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            self.train(epoch)
            if self._can_eval(epoch):
                self.eval(epoch)
