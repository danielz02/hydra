from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.layers import CurvesConv, LinesConv, LinesLinear, CurvesLinear

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import matplotlib

matplotlib.use('Agg')
from models.ensemble import Ensemble, Subspace


def requires_grad_(model: torch.nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if "popup_scores" in name:
            continue
        param.requires_grad_(requires_grad)


def Cosine(g1, g2):
    return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()


def Magnitude(g1):
    return (torch.sum(g1 ** 2, 1)).mean() * 2


# Utils for self-ensemble
@torch.no_grad()
def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f"conv{i}").weight


@torch.no_grad()
def get_stats(model, args):
    norms = {}
    numerators = {}
    difs = {}
    cossim = 0
    l2 = 0
    num_points = 3 if args.layer_type == "curve" else 2

    for i in range(num_points):
        norms[f"{i}"] = 0.0
        for j in range(i + 1, num_points):
            numerators[f"{i}-{j}"] = 0.0
            difs[f"{i}-{j}"] = 0.0

    for m in model.modules():
        if isinstance(m, CurvesConv) or isinstance(m, LinesConv):
            for i in range(num_points):
                vi = get_weight(m, i)
                norms[f"{i}"] += vi.pow(2).sum()
                for j in range(i + 1, num_points):
                    vj = get_weight(m, j)
                    numerators[f"{i}-{j}"] += (vi * vj).sum()
                    difs[f"{i}-{j}"] += (vi - vj).pow(2).sum()

    for i in range(num_points):
        for j in range(i + 1, num_points):
            cossim += numerators[f"{i}-{j}"].pow(2) / (
                norms[f"{i}"] * norms[f"{j}"]
            )
            l2 += difs[f"{i}-{j}"]

    l2 = l2.pow(0.5)
    cossim = cossim
    return cossim, l2


def cosine_diversity(model, args):
    assert args.layer_type in ["curve", "line"] and args.beta_div and args.beta_div > 0
    num_points = 3 if args.layer_type == "curve" else 2
    out = np.random.choice([i for i in range(num_points)], 2)

    i, j = out[0], out[1]
    num = 0.0
    normi = 0.0
    normj = 0.0
    for m in model.modules():
        if isinstance(m, CurvesConv) or isinstance(m, LinesConv):
            vi = get_weight(m, i)
            vj = get_weight(m, j)
            num += (vi * vj).sum()
            normi += vi.pow(2).sum()
            normj += vj.pow(2).sum()

    loss = args.beta_div * (torch.pow(num, 2) / (normi * normj))
    return loss


@torch.no_grad()
def grad_l2(model: torch.nn.Module, device):
    parameters = [p for p in model.parameters() if p.grad is not None]
    parameters_grad = [p.grad.detach() for p in parameters]
    for g in parameters_grad:
        if torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)):
            break
    return torch.norm(torch.stack([torch.norm(g, 2).to(device) for g in parameters_grad]), 2).detach().cpu().item()


def stability_loss(noisy_logits, clean_logits):
    return (1.0 / len(noisy_logits)) * nn.KLDivLoss(size_average=False)(
        F.log_softmax(noisy_logits, dim=1),
        F.softmax(clean_logits, dim=1),
    )


def subspace_diversity(models: Subspace | Ensemble, batch, alphas, device_, args):
    models.eval()
    noisy_input, targets = batch

    softma = nn.Softmax(1)
    pred = []
    margin = []
    targets = targets.to(device_)
    noisy_input.requires_grad = True
    for j in range(args.num_models):
        cur_input = noisy_input
        if isinstance(models, Subspace):
            models.fixed_alpha = alphas[j]
            output = models(cur_input)
        else:
            output = models.models[j](cur_input)

        _, predicted = output.max(1)
        pred.append(predicted == targets)
        predicted = softma(output.sort()[0])
        predicted = predicted[:, -1] - predicted[:, -2]

        grad_outputs = torch.ones(predicted.shape)
        grad_outputs = grad_outputs.to(device_)

        grad = torch.autograd.grad(predicted, cur_input, grad_outputs=grad_outputs,
                                   create_graph=True, only_inputs=True)[0]

        margin.append(grad.view(grad.size(0), -1))

    lhsloss, N = torch.tensor(0., device=device_), 0
    mse = nn.MSELoss(reduce=False)
    for ii in range(args.num_models):
        for j in range(ii + 1, args.num_models):
            flg = (pred[ii] & pred[j]).type(torch.FloatTensor).to(device_)
            grad_norm = torch.sum(mse(margin[ii], -margin[j]), dim=1)
            lhsloss += torch.sum(grad_norm * flg)
            N += torch.sum(flg)

    lhsloss /= max(N, 1)

    models.train()
    return lhsloss.item()


def update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for data in loader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        if device is not None:
            data = data.to(device)

        model(data)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
