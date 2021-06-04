# import
from torch.utils.data.dataset import random_split
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from torchvision.datasets import DatasetFolder
from os.path import join
from torchtext.utils import download_from_url, extract_archive
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# class


class TextFolder(DatasetFolder):
    def __init__(self, root):
        super().__init__(root, extensions=('.txt'), loader=None)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        with open(filepath, 'r') as f:
            text = f.readline()
        return text, label


class IMDB(Dataset):
    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.split = split
        self._download_data()
        self.class_to_idx = {k: idx for idx,
                             k in enumerate(sorted(['neg', 'pos']))}

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
        return data, label


class AG_NEWS(Dataset):
    def __init__(self, root, split) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self._download_data()
        self.class_to_idx = {'World': 0, 'Sports': 1,
                             'Business': 2, 'Sci/Tech': 3}

    def _download_data(self):
        URL = {'train': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
               'test': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv", }
        MD5 = {'train': "b1a00f826fdfbd249f79597b59e1dc12",
               'test': "d52ea96a97a2d943681189a97654912d"}
        path = download_from_url(URL[self.split], root=self.root,
                                 path=join(self.root, self.split + ".csv"),
                                 hash_value=MD5[self.split],
                                 hash_type='md5')
        self.samples = pd.read_csv(
            path, names=['label', 'title', 'description'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, title, description = self.samples.loc[index]
        return description, label-1


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters

    def prepare_data(self) -> None:
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = TextFolder(
                    root=join(self.project_parameters.data_path, stage))
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files, len(
                        self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
            if self.project_parameters.max_files is not None:
                assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset['train'].dataset.class_to_idx, self.project_parameters.classes)
            else:
                assert self.dataset['train'].class_to_idx == self.project_parameters.classes, 'the classes is not the same. please check the classes of data. from ImageFolder: {} from argparse: {}'.format(
                    self.dataset[stage].class_to_idx, self.project_parameters.classes)
        else:
            train_set = eval('{}(root=self.project_parameters.data_path, split="train")'.format(
                self.project_parameters.predefined_dataset))
            test_set = eval('{}(root=self.project_parameters.data_path, split="test")'.format(
                self.project_parameters.predefined_dataset))
            # modify the maximum number of files
            for v in [train_set, test_set]:
                if self.project_parameters.predefined_dataset == 'AG_NEWS':
                    v.samples = v.samples.iloc[np.random.permutation(v.samples.index)].reset_index(
                        drop=True)[:self.project_parameters.max_files]
                elif self.project_parameters.predefined_dataset == 'IMDB':
                    v.samples = list(np.random.permutation(v.samples))[
                        :self.project_parameters.max_files]
            train_val_lengths = [round((1-self.project_parameters.val_size)*len(train_set)),
                                 round(self.project_parameters.val_size*len(train_set))]
            train_set, val_set = random_split(
                dataset=train_set, lengths=train_val_lengths)
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}
            # get the classes from the train_set
            self.project_parameters.classes = self.dataset['train'].dataset.class_to_idx

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()
