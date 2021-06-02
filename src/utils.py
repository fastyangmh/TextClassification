# import
from ruamel.yaml import safe_load
import torch
from os.path import basename, dirname
import torch.nn as nn
import transformers
import pandas as pd

# def


def load_yaml(filepath):
    with open(file=filepath, mode='r') as f:
        config = safe_load(f)
    return config


def get_class_from_file(filepath, class_name):
    import sys
    sys.path.append('{}'.format(dirname(filepath)))
    exec('from {} import {}'.format(
        basename(filepath).split('.')[0], class_name))
    return eval('{}()'.format(class_name))


def load_checkpoint(model, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    if model.loss_function.weight is None:
        # delete the loss_function.weight in the checkpoint, because this key does not work while loading the model.
        del checkpoint['state_dict']['loss_function.weight']
    else:
        # assign the new loss_function weight to the checkpoint
        checkpoint['state_dict']['loss_function.weight'] = model.loss_function.weight
    model.load_state_dict(checkpoint['state_dict'])
    return model

# class


class CreateTransformersSequenceModel:
    def __init__(self) -> None:
        self._get_content_from_url()

    def _get_content_from_url(self):
        url = 'https://huggingface.co/transformers/pretrained_models.html'
        self.content = pd.read_html(url)[0]

    def list_models(self, architecture=None):
        if architecture is None:
            return list(self.content['Model id'])
        else:
            return list(self.content['Model id'][self.content['Architecture'].str.lower() == architecture.lower()])

    def create_tokenizer(self, model_id):
        if model_id in self.list_models():
            architecture, model_id, detail = self.content[self.content['Model id']
                                                          == model_id].values[0]
            tokenizer_class_name = [v for v in dir(transformers) if v.lower(
            ).startswith('{}Tokenizer'.format(architecture).lower())][-1]
            tokenizer = eval('transformers.{}.from_pretrained("{}")'.format(
                tokenizer_class_name, model_id))
            return tokenizer
        else:
            assert False, 'the model_id does not exist in pretrained models. the model_id is {}'.format(
                model_id)

    def create_model(self, model_id, num_classes):
        if model_id in self.list_models():
            architecture, model_id, detail = self.content[self.content['Model id']
                                                          == model_id].values[0]
            print(detail)
            model_class_name = [v for v in dir(transformers) if '{}ForSequenceClassification'.format(
                architecture).lower() == v.lower()]
            assert model_class_name, 'the model_id does not apply to sequence classification. the model_id is {}'.format(
                model_id)
            model = eval('transformers.{}.from_pretrained("{}")'.format(
                model_class_name[-1], model_id))
            if model.classifier.out_features != num_classes:
                model.classifier = nn.Linear(
                    in_features=model.classifier.in_features, out_features=num_classes)
            return model
        else:
            assert False, 'the model_id does not exist in pretrained models. the model_id is {}'.format(
                model_id)
