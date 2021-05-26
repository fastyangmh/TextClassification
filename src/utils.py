# import
from ruamel.yaml import safe_load

# def
def load_yaml(filepath):
    with open(file=filepath, mode='r') as f:
        config = safe_load(f)
    return config
