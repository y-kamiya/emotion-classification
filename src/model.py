from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification


class BertModel():
    @classmethod
    def create(cls, config, n_labels):
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking' if config.lang == 'ja' else 'bert-base-uncased'

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=n_labels, return_dict=True)
        if config.freeze_base_model:
            for param in model.base_model.parameters():
                param.requires_grad = False

        if config.custom_head:
            model.classifier = CustomClassificationHead(config, model.config.hidden_size, n_labels)

        tokenizer = BertTokenizer.from_pretrained(model_name, padding=True)

        return model.to(config.device), tokenizer


class CustomClassificationHead(nn.Module):
    def __init__(self, config, input_dim, n_labels):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, n_labels)
        self.dropout = nn.Dropout(p=0.3)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        # dropout is applied before this method is called
        # https://github.com/huggingface/transformers/blob/v4.1.1/src/transformers/models/bert/modeling_bert.py#L1380
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(self.dropout(x)))
        x = self.prelu3(self.fc3(self.dropout(x)))
        return self.fc4(self.dropout(x))


