import csv
import io
import os

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config


class TextClassificationDataset(BaseDataset):
    def __init__(self, config, phase):
        super(TextClassificationDataset, self).__init__(config, phase)

        self.label_index_map = self.create_label_index_map()

        self.n_labels = len(self.label_index_map)

        data_file = config.data_file
        if data_file is None:
            data_file = f"{phase}.tsv"

        filepath = os.path.join(config.dataroot, data_file)
        self.texts = []
        self.labels = torch.empty(0)
        with io.open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                text = "" if len(row) == 0 else row[0]
                label_name = "none" if len(row) <= 1 else row[1]
                if label_name not in self.label_index_map and phase != "predict":
                    self.config.logger.warning(
                        f"{label_name} is invalid label name, skipped. text: {text}"
                    )
                    continue

                self.texts.append(text)

                labels = torch.zeros(1, self.n_labels)
                if label_name == "none":
                    assert (
                        self.config.predict
                    ), "label is necessary not when args.predict == true"
                else:
                    index = self.label_index_map[label_name]
                    labels[0][index] = 1

                self.labels = torch.cat([self.labels, labels])

    def create_label_index_map(self):
        label_set = set()

        label_file = self.config.label_file
        if label_file is None:
            label_file = "labels.txt"

        filepath = os.path.join(self.config.dataroot, label_file)
        with io.open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                label_set.add(row[0])

        return {label: i for i, label in enumerate(sorted(list(label_set)))}

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


class EmotionDataset(TextClassificationDataset):
    def __init__(self, config, phase):
        super(EmotionDataset, self).__init__(config, phase)

    def create_label_index_map(self):
        return {
            "anger": 0,
            "disgust": 1,
            "joy": 2,
            "sadness": 3,
            "surprise": 4,
        }


class SemEval2018EmotionDataset(BaseDataset):
    label_index_map = {
        "anger": 0,
        "anticipation": 1,
        "disgust": 2,
        "fear": 3,
        "joy": 4,
        "love": 5,
        "optimism": 6,
        "pessimism": 7,
        "sadness": 8,
        "surprise": 9,
        "trust": 10,
    }

    def __init__(self, config, phase):
        self.config = config
        self.n_labels = len(self.label_index_map)

        filepath = os.path.join(config.dataroot, f"{phase}.tsv")
        self.texts = []
        self.labels = torch.empty(0)
        with io.open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.texts.append(row[1])

                labels = torch.zeros(1, self.n_labels)
                for i in self.label_index_map.values():
                    column_index = i + 2
                    labels[0][i] = int(row[column_index])

                self.labels = torch.cat([self.labels, labels])

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)
