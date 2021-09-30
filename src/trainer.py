import sys
import os
import io
import time
import uuid
import argparse
import apex
import torch
from torch.nn import functional as F
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
from transformers import AdamW
from logzero import setup_logger
from sklearn import metrics
import pycm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tabulate import tabulate
from tqdm import tqdm

from dataset import EmotionDataset, SemEval2018EmotionDataset, TextClassificationDataset
from model import BertModel, RobertaModel


class Trainer:
    def __init__(self, config):
        self.config = config

        if self.config.predict:
            dataset = self.config.dataset_class(self.config, 'predict')
            self.dataloader_predict = DataLoader(dataset, batch_size=self.config.batch_size)
        else:
            dataset = self.config.dataset_class(self.config, 'train')
            self.dataloader_train = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            data_eval= self.config.dataset_class(self.config, 'eval')
            self.dataloader_eval = DataLoader(data_eval, batch_size=self.config.batch_size, shuffle=False)

        self.model, self.tokenizer = self.__create_model(config, len(dataset.label_index_map))

        self.optimizer = optim.RAdam(self.model.parameters(), lr=config.lr)

        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        self.best_f1_score = 0.0

        if config.fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, 'O1')

        self.load(self.config.model_path)

        if self.config.freeze_base:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def __create_model(self, config, n_labels):
        if config.model_type == 'bert':
            return BertModel.create(config, n_labels)

        if config.model_type == 'roberta':
            return RobertaModel.create(config, n_labels)

        assert False, f'model_type: {config.model_type} is not defined'

    def forward(self, inputs, labels):
        position_ids = None
        if self.config.model_type == 'roberta':
            batch_size, token_size = inputs['input_ids'].shape
            position_ids = torch.arange(token_size).expand((batch_size, -1)).to(self.config.device)

        if self.config.multi_labels:
            outputs = self.model(**inputs, position_ids=position_ids)
            loss = F.binary_cross_entropy_with_logits(outputs.logits, labels)
            return loss, 0 < outputs.logits

        outputs = self.model(**inputs, labels=torch.argmax(labels, dim=1), position_ids=position_ids)
        return outputs.loss, torch.argmax(outputs.logits, dim=1)

    def train(self, epoch):
        self.model.train()

        for i, (texts, labels) in enumerate(self.dataloader_train):
            start_time = time.time()
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
            labels = labels.to(self.config.device)

            loss, _ = self.forward(inputs, labels)

            if self.config.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if i % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.config.logger.info('train epoch: {}, step: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, i, loss, elapsed_time))

            self.writer.add_scalar('loss/train', loss, epoch, start_time)
            mlflow.log_metric('loss/train', loss.item(), epoch)

        self.save(self.config.model_path)

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()

        all_labels = torch.empty(0)
        all_preds = torch.empty(0)
        losses = []
        start_time = time.time()

        for i, (texts, labels) in enumerate(self.dataloader_eval):
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
            labels = labels.to(self.config.device)

            loss, preds = self.forward(inputs, labels)

            losses.append(loss)

            if not self.config.multi_labels:
                labels = torch.argmax(labels, dim=1)

            all_labels = torch.cat([all_labels, labels.cpu()])
            all_preds = torch.cat([all_preds, preds.cpu()])

        elapsed_time = time.time() - start_time
        average_loss = sum(losses)/len(losses)
        self.config.logger.info('eval epoch: {}, loss: {:.2f}, time: {:.2f}'.format(epoch, average_loss, elapsed_time))

        self.__log_confusion_matrix(all_preds, all_labels, epoch)

        df = pd.DataFrame(metrics.classification_report(all_labels, all_preds, output_dict=True))
        print(tabulate(df, headers='keys', tablefmt="github", floatfmt='.3f'))

        if not self.config.eval_only:
            f1_score = df.loc['f1-score']
            micro = f1_score['micro avg'] if 'micro avg' in f1_score else f1_score['accuracy']
            macro = f1_score['macro avg']
            mlflow.log_metric('loss/eval', average_loss.item(), epoch)
            mlflow.log_metric('metrics/f1_score_micro', micro, epoch)
            mlflow.log_metric('metrics/f1_score_macro', macro, epoch)
            self.writer.add_scalar('loss/eval', average_loss, epoch, start_time)
            self.writer.add_scalar('metrics/f1_score_micro(accuracy)', micro, epoch, start_time)
            self.writer.add_scalar('metrics/f1_score_macro', macro, epoch, start_time)

            if self.best_f1_score < micro:
                self.best_f1_score = micro
                self.save(self.config.best_model_path)

    @torch.no_grad()
    def predict(self):
        label_map = {value: key for key, value in self.dataloader_predict.dataset.label_index_map.items()}
        label_map[-1] = 'none'
        np.set_printoptions(precision=0)

        output_path = os.path.join(self.config.dataroot, 'predict_result')
        with open(output_path, 'w') as f:
            for i, (texts, labels) in tqdm(enumerate(self.dataloader_predict)):
                if not self.config.multi_labels:
                    labels = torch.tensor([-1 if sum(onehot) == 0 else torch.argmax(onehot) for onehot in labels])
                inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.config.device)
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                probs = F.softmax(outputs.logits, dim=1) * 100
                for j in range(len(texts)):
                    pred_label_name = label_map[preds[j].item()]
                    true_label_name = label_map[labels[j].item()]
                    prob = probs[j].cpu().numpy()
                    f.write(f'{pred_label_name}\t{prob}\t{true_label_name}\t{texts[j]}\n')

        self.config.logger.info(f'write predicted result to {output_path}')

    def __log_confusion_matrix(self, all_preds, all_labels, epoch):
        buf = io.BytesIO()
        dataset = self.dataloader_eval.dataset
        label_map = {value: key for key, value in dataset.label_index_map.items()}

        np.set_printoptions(precision=3)

        if self.config.multi_labels:
            fig, axes = plt.subplots(1, len(label_map.keys()), figsize=(25, 5))
            cm = metrics.multilabel_confusion_matrix(y_pred=all_preds.numpy(), y_true=all_labels.numpy())
            for i in range(len(label_map.keys())):
                mat = np.array([[cm[i][1][1], cm[i][1][0]], [cm[i][0][1], cm[i][0][0]]])
                result = mat / mat.sum(axis=1, keepdims=True)
                print(f'{label_map[i]}\n{result}\n')

                display = metrics.ConfusionMatrixDisplay(result, display_labels=['P', 'N'])
                display.plot(ax=axes[i], cmap=plt.cm.Blues, values_format='.2f')
                display.ax_.set_title(label_map[i])
                display.ax_.set_ylabel('True label' if i == 0 else '')
                display.ax_.set_yticklabels(['P', 'N'] if i == 0 else [])
                display.im_.colorbar.remove()

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.colorbar(display.im_, ax=axes)
            plt.savefig(buf, format="png", dpi=180)
        else:
            cm = metrics.confusion_matrix(y_pred=all_preds.numpy(), y_true=all_labels.numpy(), normalize='true')
            display = metrics.ConfusionMatrixDisplay(cm, display_labels=label_map.values())
            display.plot(cmap=plt.cm.Blues)
            display.figure_.savefig(buf, format="png", dpi=180)
            
            cm = pycm.ConfusionMatrix(actual_vector=all_labels.numpy(), predict_vector=all_preds.numpy())
            cm.relabel(mapping=label_map)
            cm.print_normalized_matrix()

        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite('confusion.png', img)
        mlflow.log_artifact('confusion.png')
        self.writer.add_image('confusion_maatrix', img, epoch, dataformats='HWC')

    def save(self, model_path):
        if self.config.no_save:
            return

        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': apex.amp.state_dict() if self.config.fp16 else None,
            'batch_size': self.config.batch_size,
            'fp16': self.config.fp16,
        }
        torch.save(data, model_path)
        self.config.logger.info(f'save model to {model_path}')

    def load(self, model_path):
        if not os.path.isfile(model_path):
            self.config.logger.warning(f'model_path: {model_path} is not found')
            return

        data = torch.load(model_path, map_location=self.config.device_name)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        if self.config.fp16:
            apex.amp.load_state_dict(data['amp'])

        self.config.logger.info(f'load model from {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--loglevel', default='DEBUG')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--n_labels', type=int, default=5, help='number of classes to train')
    parser.add_argument('--dataroot', default='data', help='path to data directory')
    parser.add_argument('--data_file', default=None, help='file name with data')
    parser.add_argument('--label_file', default=None, help='file name with labels')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
    parser.add_argument('--epochs', type=int, default=10, help='epoch count')
    parser.add_argument('--fp16', action='store_true', help='run model with float16')
    parser.add_argument('--lang', default='ja', choices=['en', 'ja'])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('--freeze_base', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--multi_labels', action='store_true')
    parser.add_argument('--dataset_class_name', default='EmotionDataset', choices=['EmotionDataset', 'SemEval2018EmotionDataset', 'TextClassificationDataset'])
    parser.add_argument('--model_type', default='roberta', choices=['bert', 'roberta'])
    parser.add_argument('--custom_head', action='store_true')
    parser.add_argument('--freeze_base_model', action='store_true')
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

    args.tensorboard_log_dir = f'{args.dataroot}/runs/{args.name}'

    args.model_path = f'{args.dataroot}/{args.name}.pth'
    args.best_model_path = f'{args.dataroot}/{args.name}.best.pth'

    args.dataset_class = globals()[args.dataset_class_name]
    if args.dataset_class_name == 'SemEval2018EmotionDataset':
        args.multi_labels = True

    logger.info(args)
    mlflow.set_tracking_uri(f'{args.dataroot}/mlruns')
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
