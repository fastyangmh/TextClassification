# import
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# class


class AutoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.backbone_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-cased')
        self.backbone_model.classifier = nn.Linear(
            in_features=self.backbone_model.classifier.in_features, out_features=2)

    def forward(self, x):
        x = self.tokenizer(list(x), padding='max_length',
                           max_length=128, return_tensors='pt')
        return self.backbone_model(**x).logits


if __name__ == '__main__':
    # create model
    model = AutoModel()

    # create input
    x = ['Hello World!']*10

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(len(x))
    print(y.shape)
