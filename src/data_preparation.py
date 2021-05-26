# import
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from torchvision.datasets import DatasetFolder
from os.path import join
from transformers import DistilBertTokenizerFast
from torchtext.utils import download_from_url, extract_archive
from torch.utils.data import Dataset
import pandas as pd

# global variables
TOKENIZER = {
    'DistilBert': "DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')"}

# class


class TextFolder(DatasetFolder):
    def __init__(self, root, project_parameters, max_length, transform=None):
        super().__init__(root, extensions=('.txt'), transform=transform, loader=None)
        self.project_parameters = project_parameters
        self.tokenizer = eval(TOKENIZER[project_parameters.backbone_model])
        self.max_length = max_length

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        with open(filepath, 'r') as f:
            text = f.readline()
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"), label


class MyIMDB(Dataset):
    def __init__(self, root, split, max_length):
        super().__init__()
        self.root = root
        self.split = split
        self._download_data()
        self.tokenizer = eval(TOKENIZER[project_parameters.backbone_model])
        self.class_to_idx = {k: idx for idx,
                             k in enumerate(sorted(['neg', 'pos']))}
        self.max_length = max_length

    def _download_data(self):
        URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
        dataset_tar = download_from_url(
            URL, root=self.root, hash_value=MD5, hash_type='md5')
        extracted_files = extract_archive(dataset_tar)
        samples = []
        for fname in extracted_files:
            if 'urls' in fname:
                continue
            elif self.split in fname and ('pos' in fname or 'neg' in fname):
                with open(fname, encoding="utf8") as f:
                    label = 'pos' if 'pos' in fname else 'neg'
                    text = f.readline()
                samples.append((text, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, label = self.samples[index]
        label = self.class_to_idx[label]
        return self.tokenizer(data, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"), label


class MyAG_NEWS(Dataset):
    def __init__(self, root, split, max_length) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self._download_data()
        self.tokenizer = eval(TOKENIZER[project_parameters.backbone_model])
        self.class_to_idx = {'World': 1, 'Sports': 2,
                             'Business': 3, 'Sci/Tech': 4}
        self.max_length = max_length

    def _download_data(self):
        URL = {'train': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
               'test': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", }
        MD5 = {'train': "b1a00f826fdfbd249f79597b59e1dc12",
               'test': "d52ea96a97a2d943681189a97654912d"}
        path = download_from_url(URL[self.split], root=self.root,
                                 path=join(self.root, self.split + ".csv"),
                                 hash_value=MD5[self.split],
                                 hash_type='md5')
        self.samples = pd.read_csv(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, title, description = self.samples.loc[index]
        return self.tokenizer(description, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"), label


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()
