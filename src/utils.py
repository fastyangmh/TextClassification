# import
from ruamel.yaml import safe_load
import torch
from os.path import basename, dirname

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
