# import
import argparse
from os import makedirs
from posixpath import join
import torch
from src.utils import load_yaml
from os.path import abspath

# class


class ProjectParameters:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base
        self._parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune', 'evaluate'], required=True,
                                  help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self._parser.add_argument(
            '--data_path', type=str, required=True, help='the data path.')
        self._parser.add_argument('--predefined_dataset', type=str, default=None, choices=[
                                  'AG_NEWS', 'IMDB'], help='the predefined dataset that provided the AG_NEWS and IMDB datasets.')
        self._parser.add_argument(
            '--random_seed', type=self._str_to_int, default=0, help='the random seed.')
        self._parser.add_argument(
            '--save_path', type=str, default='save/', help='the path which stores the checkpoint of PyTorch lightning.')
        self._parser.add_argument('--no_cuda', action='store_true', default=False,
                                  help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self._parser.add_argument('--gpus', type=self._str_to_int_list, default=-1,
                                  help='number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node. if give -1 will use all available GPUs.')
        self._parser.add_argument(
            '--parameters_config_path', type=str, default=None, help='the parameters config path.')

        # data preparation
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')
        self._parser.add_argument('--classes', type=self._str_to_str_list, required=True,
                                  help='the classes of data. if use a predefined dataset, please set value as None.')
        self._parser.add_argument('--val_size', type=float, default=0.1,
                                  help='the validation data size used for the predefined dataset.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        self._parser.add_argument(
            '--no_balance', action='store_true', default=False, help='whether to balance the data.')
        self._parser.add_argument('--max_length', type=int, default=512,
                                  help='pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.')

        # model
        self._parser.add_argument('--backbone_model', type=str, choices=[
                                  'DistilBert'], required=True, help='if you want to use a self-defined model, give the path of the self-defined model. otherwise, the provided backbone model is as followed by the transformers packages.')

        # debug
        self._parser.add_argument(
            '--max_files', type=self._str_to_int, default=None, help='the maximum number of files for loading files.')

    def _str_to_int(self, s):
        return None if s == 'None' or s == 'none' else int(s)

    def _str_to_int_list(self, s):
        return [int(v) for v in s.split(',') if len(v) > 0]

    def _str_to_str_list(self, s):
        return [str(v) for v in s.split(',') if len(v) > 0]

    def _get_new_dict(self, old_dict, yaml_dict):
        for k in yaml_dict.keys():
            del old_dict[k]
        return {**old_dict, **yaml_dict}

    def parse(self):
        project_parameters = self._parser.parse_args()
        if project_parameters.parameters_config_path is not None:
            project_parameters = argparse.Namespace(**self._get_new_dict(old_dict=vars(
                project_parameters), yaml_dict=load_yaml(filepath=abspath(project_parameters.parameters_config_path))))
        else:
            del project_parameters.parameters_config_path

        # base
        project_parameters.data_path = abspath(
            path=project_parameters.data_path)
        if project_parameters.predefined_dataset is not None:
            # the classes of predefined dataset will automatically get from data_preparation
            project_parameters.data_path = join(
                project_parameters.data_path, project_parameters.predefined_dataset)
            makedirs(project_parameters.data_path, exist_ok=True)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda
        project_parameters.gpus = project_parameters.gpus if project_parameters.use_cuda else 0

        # data preparation
        if project_parameters.predefined_dataset is not None:
            if project_parameters.predefined_dataset == 'AG_NEWS':
                project_parameters.num_classes = 4
            elif project_parameters.predefined_dataset == 'IMDB':
                project_parameters.num_classes = 2
            else:
                assert False, 'please check the predefined dataset. the predefined dataset: {}'.format(
                    project_parameters.predefined_dataset)
        else:
            project_parameters.classes = {
                c: idx for idx, c in enumerate(sorted(project_parameters.classes))}
            project_parameters.num_classes = len(project_parameters.classes)
        project_parameters.use_balance = not project_parameters.no_balance and project_parameters.predefined_dataset is None

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for name, value in vars(project_parameters).items():
        print('{:<20}= {}'.format(name, value))
