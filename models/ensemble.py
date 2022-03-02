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


class BezierCurve(Ensemble):
    def __init__(self, subspace_model, is_layerwise: bool):
        models = [BezierEndPoint(subspace_model, i) for i in range(3)]
        super().__init__(models)

        self.layerwise = is_layerwise
        self.subspace_model = subspace_model

    def __sample_subnet(self, **kwargs):
        fix_alpha = kwargs.get("fixed_alpha")
        if fix_alpha is None and not self.training and not hasattr(self, "fixed_alpha"):
            fix_alpha = 0.5
        elif hasattr(self, "fixed_alpha"):
            fix_alpha = self.fixed_alpha

        if fix_alpha:
            print(f"Using fixed alpha {fix_alpha}")

        alpha = np.random.uniform(0, 1) if not fix_alpha else fix_alpha
        for m in self.subspace_model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                setattr(m, f"alpha", alpha)
            if self.layerwise:
                alpha = np.random.uniform(0, 1) if not fix_alpha else fix_alpha

    def forward(self, x):
        self.__sample_subnet()
        return self.subspace_model(x)


class BezierEndPoint(nn.Module):
    def __init__(self, bezier_subspace, idx: int):
        assert idx in [0, 1, 2]
        super().__init__()

        self.point_idx = idx
        self.bezier_subspace = bezier_subspace

    def forward(self, x):
        setattr(self.bezier_subspace, f"w{self.point_idx}", True)
        out = self.bezier_subspace(x)
        delattr(self.bezier_subspace, f"w{self.point_idx}")

        return out
