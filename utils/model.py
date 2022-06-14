import sys
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import models

import os
import math
import numpy as np

from models.ensemble import Subspace, Ensemble
from models.layers import SubnetConv, SubnetLinear, CurvesConv, CurvesLinear, CurvesBN, LinesConv, LinesBN, LinesLinear


# TODO: avoid freezing bn_params
# Some utils are borrowed from https://github.com/allenai/hidden-networks
def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear, nn.BatchNorm2d
    elif layer_type == "subnet":
        return SubnetConv, SubnetLinear, nn.BatchNorm2d
    elif layer_type == "curve":
        return CurvesConv, CurvesLinear, CurvesBN
    elif layer_type == "line":
        return LinesConv, LinesLinear, LinesBN
    else:
        raise ValueError("Incorrect layer type")


def create_model(args, gpu_list, device, logger):
    ensemble_models = []
    cl, ll, bn = get_layers(args.layer_type)

    args.init_type = getattr(args, "init_type", "kaiming_normal")
    for _ in range(args.num_models):
        if args.layer_type in ["curve", "line"]:
            assert args.num_models == 3
            model = models.__dict__[args.arch](
                    cl, ll, args.init_type, num_classes=args.num_classes, args=args, bn_layer=bn
                )
        # elif args.ddp or len(gpu_list) >= 1:
        #     model = models.__dict__[args.arch](cl, ll, args.init_type, num_classes=args.num_classes, args=args)
        else:
            model = models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes, args=args
            ).to(device)

        if args.ddp:
            import horovod.torch as hvd

            device = 'cuda'
            model = model.to(device)
            is_ddp = (args.ddp and hvd.rank() == 0) or not args.ddp
        else:
            print("Using DataParallel")
            model = nn.parallel.DataParallel(model, gpu_list).to(device)
            is_ddp = True

        if is_ddp and logger:
            logger.info(model)

        # Customize models for training/pruning/fine-tuning
        prepare_model(model, args)
        ensemble_models.append(model)
        if args.layer_type in ["curve", "line"]:
            break

    if args.layer_type in ["curve", "line"]:
        ensemble_model = Subspace(*ensemble_models, is_layerwise=args.layerwise)
    else:
        ensemble_model = Ensemble(ensemble_models)

    return ensemble_model


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")


def snip_init(model, criterion, optimizer, train_loader, device, args):
    print("Using SNIP initialization")
    assert args.exp_mode == "pretrain"
    optimizer.zero_grad()
    # init the score with kaiming normal init
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            nn.init.kaiming_normal_(m.popup_scores, mode="fan_in")

    set_prune_rate_model(model, 1.0)
    unfreeze_vars(model, "popup_scores")

    # take a forward pass and get gradients
    for _, data in enumerate(train_loader):
        images, target = data[0].to(device), data[1].to(device)

        output = model(images)
        loss = criterion(output, target)

        loss.backward()
        break

    # update scores with their respective connection sensitivty
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            print(m.popup_scores.data)
            m.popup_scores.data = m.popup_scores.grad.data.abs()
            print(m.popup_scores.data)

    # update k back to args.k.
    set_prune_rate_model(model, args.k)
    freeze_vars(model, "popup_scores")


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model, k=6):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming uniform init
            m.popup_scores.data = (
                    math.sqrt(k / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print(f"previous std = {torch.std(m.weight.data)}")
            m.weight.data = 1 / math.sqrt(k) * m.weight.data
            # print(f"new std = {torch.std(m.weight.data)}")


def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base of training mode.
    """

    set_prune_rate_model(model, args.k)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None  | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None  | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(
            f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
        )
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    if args.scores_init_type is not None:
        initialize_scores(model, args.scores_init_type)


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                    subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense


@torch.no_grad()
def subspace_to_subnet(model: Ensemble, subspace_dict: dict, alphas, subspace_type):
    assert isinstance(model, Ensemble)
    assert len(alphas) == len(model.models)
    assert subspace_type in ["curve", "line"]

    subnet_dict = OrderedDict()

    for i, alpha in enumerate(alphas):
        for k, _ in model.models[i].state_dict().items():
            if "popup_scores" in k:
                continue

            print(f"Converting: {k}")

            layer_type = k.split(".")[-2]
            k_subspace = f"subspace_model.{k}"

            if "conv" in layer_type and "weight" in k:
                v1 = subspace_dict[k_subspace]
                v2 = subspace_dict[f"{k_subspace.replace('.weight', '')}.conv1.weight"]
                if subspace_type == "curve":
                    v3 = subspace_dict[f"{k_subspace.replace('.weight', '')}.conv2.weight"]
                    v = ((1 - alpha) ** 2) * v1 + 2 * alpha * (1 - alpha) * v2 + (alpha ** 2) * v3
                else:
                    v = ((1 - alpha) * v1 + alpha * v2).clone()
            elif "fc" in layer_type and "weight" in k:
                v1 = subspace_dict[k_subspace]
                if subspace_type == "curve":
                    v2 = subspace_dict[f"{k_subspace.replace('.weight', '')}.conv1.weight"]
                    v3 = subspace_dict[f"{k_subspace.replace('.weight', '')}.conv2.weight"]
                    v = ((1 - alpha) ** 2) * v1 + 2 * alpha * (1 - alpha) * v2 + (alpha ** 2) * v3
                else:
                    v2 = subspace_dict[f"{k_subspace.replace('.weight', '')}.linear1.weight"]
                    v = ((1 - alpha) * v1 + alpha * v2).clone()
            elif "bn" in layer_type and ("weight" in k or "bias" in k):
                v1 = subspace_dict[k_subspace]
                v2 = subspace_dict[f"{k_subspace}1"]
                if subspace_type == "curve":
                    v3 = subspace_dict[f"{k_subspace}2"]
                    v = ((1 - alpha) ** 2) * v1 + 2 * alpha * (1 - alpha) * v2 + (alpha ** 2) * v3
                else:
                    v = ((1 - alpha) * v1 + alpha * v2).clone()
            else:
                v = subspace_dict[k_subspace].clone()
            subnet_dict[f"models.{i}.{k}"] = v

    missing_keys, unexpected_keys = model.load_state_dict(subnet_dict, strict=False)
    print(f"Subspace converted!\nMissing keys {missing_keys}\n Unexpected keys {unexpected_keys}", file=sys.stderr)

    return subnet_dict


@torch.no_grad()
def ckpt_combine(model: Ensemble, ckpt: List[dict]):
    """
        Combine multiple checkpoint file to an ensemble checkpoint
    """
    model_dict = OrderedDict()
    for k, _ in model.state_dict().items():
        if "pop" in k:
            continue
        _, i, *_ = k.split(".")
        model_dict[k] = ckpt[int(i)]["state_dict"][k.replace(f"models.{i}.1", "1.module")]
    return model_dict


def dense_to_subnet(model, state_dict):
    """
        Load a dict with dense-layer in a model trained with subnet layers. 
    """
    model.load_state_dict(state_dict, strict=False)


def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for i, v in model.named_modules():
            if isinstance(v, (nn.Conv2d, nn.Linear)):
                if i + ".weight" in state_dict.keys():
                    d = state_dict[i + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    if verbose:
                        print(i, v, p)
        return np.mean(pl)


def sanity_check_paramter_updates(model, last_ckpt, layer_type):
    """
        Check whether weights/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        if hasattr(v, "weight") and (layer_type == "dense" or hasattr(v, "popup_scores")):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if (layer_type == "subnet" or layer_type == "curve") and getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2) if layer_type == "subnet" else True
