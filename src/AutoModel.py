# import
import torch.nn as nn
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# class


class AutoModelTokenizer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def __call__(self, text, *args: Any, **kwds: Any):
        return self.tokenizer(text, padding='max_length', max_length=128, return_tensors='pt')


class AutoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-cased')
        self.backbone_model.classifier = nn.Linear(
            in_features=self.backbone_model.classifier.in_features, out_features=10)

    def forward(self, *args: Any, **kwds: Any):
        return self.backbone_model(*args, **kwds)


if __name__ == '__main__':
    # create model
    model = AutoModel()

    # create tokenizer
    tokenizer = AutoModelTokenizer()

    # create input
    x = tokenizer(text='Hello World!')

    # get model output
    y = model(**x)

    # display the dimension of input and output
    for k in x.keys():
        print(x[k].shape)
    print(y)
