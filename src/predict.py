# import
from torch.utils.data.dataloader import DataLoader
from src.data_preparation import TextFolder
import torch
from src.model import create_model
from src.project_parameters import ProjectParameters
import numpy as np
import torch.nn.functional as F

# class


class Predict:
    def __init__(self, project_parameters) -> None:
        self.project_parameters = project_parameters
        self.model = create_model(project_parameters=project_parameters).eval()

    def get_result(self, data_path):
        result = []
        if '.txt' in data_path:
            with open(data_path, 'r') as f:
                data = f.readline()
            with torch.no_grad():
                pred = self.model(list([data]))
                if '.py' in self.project_parameters.backbone_model:
                    pred = F.softmax(pred, dim=-1)
                result.append(pred.tolist()[0])
        else:
            dataset = TextFolder(root=data_path)
            data_loader = DataLoader(dataset=dataset, batch_size=self.project_parameters.batch_size,
                                     pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)
            with torch.no_grad():
                for data, _ in data_loader:
                    result.append(self.model(data).tolist())
        return np.concatenate(result, 0).round(2)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict the data path
    result = Predict(project_parameters=project_parameters).get_result(
        data_path=project_parameters.data_path)
    # use [:-1] to remove the latest comma
    print(('{},'*project_parameters.num_classes).format(*
                                                        project_parameters.classes.keys())[:-1])
    print(result)
