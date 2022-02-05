import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += model(x)
            output = outputs / len(self.models)
            return output
        else:
            return self.models[0](x)
