import numpy as np
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


class BezierCurve(nn.Module):
    def __init__(self, subspace_model, is_layerwise: bool):
        super().__init__()

        self.layerwise = is_layerwise
        self.subspace_model = subspace_model

    def __sample_subnet(self, **kwargs):
        fix_alpha = kwargs.get("fixed_alpha")
        if fix_alpha is None and not self.training and not hasattr(self, "fixed_alpha"):
            fix_alpha = 0.5
        elif hasattr(self, "fixed_alpha"):
            fix_alpha = self.fixed_alpha

        alpha = np.random.uniform(0, 1) if not fix_alpha else fix_alpha
        for m in self.subspace_model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                setattr(m, f"alpha", alpha)
            if self.layerwise:
                alpha = np.random.uniform(0, 1) if not fix_alpha else fix_alpha

    def forward(self, x):
        self.__sample_subnet()
        return self.subspace_model(x)


class SubspaceEnsemble(Ensemble):
    def __init__(self, subspace, num_models=3):
        super().__init__([subspace] * num_models)
        self.alpha = np.random.uniform(size=num_models)

        assert len(self.models) == len(self.alpha)

    def reset_alpha(self):
        self.alpha = np.random.uniform(size=self.num_models)

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for alpha, model in zip(self.alpha, self.models):
                model.fixed_alpha = alpha
                outputs += model(x)
            output = outputs / len(self.models)
            return output
        else:
            return self.models[0](x)
