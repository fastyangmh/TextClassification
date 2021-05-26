# import
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from torchvision.datasets import DatasetFolder
from os.path import join
from transformers import DistilBertTokenizerFast

# global variables
TOKENIZER = {
    'DistilBert': "DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')"}

# class


class TextFolder(DatasetFolder):
    def __init__(self, root, project_parameters, transform=None):
        super().__init__(root, extensions=('.txt'), transform=transform, loader=None)
        self.project_parameters = project_parameters
        self.tokenizer = eval(TOKENIZER[project_parameters.backbone_model])

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        with open(filepath, 'r') as f:
            text = f.readline()
        return self.tokenizer(text), label


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()
