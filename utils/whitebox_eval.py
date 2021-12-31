import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os

import data
import models
from utils.model import get_layers, prepare_model

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack
from advertorch.attacks.utils import attack_whole_dataset
from models.ensemble import Ensemble
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("arch", type=str)
parser.add_argument('dataset', type=str, choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet"))
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("attack_type", type=str, help="choose from [fgsm, pgd, mim, bim, jsma, cw, ela]")
parser.add_argument("layer_type", type=str)
parser.add_argument("num_classes", type=int)
parser.add_argument('--exp-mode', type=str, required=True)
parser.add_argument('--k', type=float, required=True)
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--num-models', type=int, required=True)
parser.add_argument('--adv-eps', default=0.2, type=float)
parser.add_argument('--adv-steps', default=10, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.1, type=float)  # for jsma, cw, ela
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of images used from training set")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=256)
parser.add_argument(
    "--freeze-bn",
    action="store_true",
    default=False,
    help="freeze batch-norm parameters in pruning",
)
parser.add_argument(
    "--scores_init_type",
    choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
    default="kaiming_normal",
    help="Which init to use for relevance scores",
)

args = parser.parse_args()


def main():
    checkpoint = torch.load(args.base_classifier)
    ensemble_models = []
    for _ in range(args.num_models):
        cl, ll = get_layers(args.layer_type)
        model = models.__dict__[args.arch](
            cl, ll, "kaiming_normal", num_classes=args.num_classes, in_channels=(1 if args.dataset == "MNIST" else 3)
        ).to(f"cuda:{args.gpu}")
        prepare_model(model, args)
        model.eval()
        ensemble_models.append(model)
    ensemble = Ensemble(ensemble_models)
    ensemble = ensemble.to(f"cuda:{args.gpu}")
    ensemble.load_state_dict(checkpoint['state_dict'])
    ensemble.eval()

    print('Model loaded')

    test_dataset = data.__dict__[args.dataset](args, normalize=(args.dataset == "imagenet"))
    _, testloader = test_dataset.data_loaders()

    loss_fn = nn.CrossEntropyLoss()

    correct_or_not = []
    for i in range(args.random_start):
        print("Phase %d" % i)
        torch.manual_seed(i)
        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

        if args.attack_type == "pgd":
            adversary = LinfPGDAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        elif args.attack_type == "fgsm":
            adversary = GradientSignAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                clip_min=0., clip_max=1., targeted=False)
        elif args.attack_type == "mim":
            adversary = LinfMomentumIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
                targeted=False)
        elif args.attack_type == "bim":
            adversary = LinfBasicIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / args.steps, clip_min=0., clip_max=1.,
                targeted=False)
        elif args.attack_type == "cw":
            adversary = CarliniWagnerL2Attack(
                ensemble, confidence=0.1, max_iterations=1000, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, binary_search_steps=1, initial_const=args.coeff)

        elif args.attack_type == "ela":
            adversary = ElasticNetL1Attack(
                ensemble, initial_const=args.coeff, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10
            )
        elif args.attack_type == "jsma":
            adversary = JacobianSaliencyMapAttack(
                ensemble, clip_min=0., clip_max=1., num_classes=10, gamma=args.coeff)
        else:
            adversary = None

        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=f"cuda:{args.gpu}")

        correct_or_not.append(label == advpred)

    correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

    print("")
    if args.attack_type == "cw" or args.attack_type == "ela":
        print("%s (c = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
                                                              100. * (label == pred).sum().item() / len(label),
                                                              100. * correct_or_not.sum().item() / len(label)))
    elif args.attack_type == "jsma":
        print("%s (gamma = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
                                                                  100. * (label == pred).sum().item() / len(label),
                                                                  100. * correct_or_not.sum().item() / len(label)))
    else:
        print("%s (eps = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
                                                                100. * (label == pred).sum().item() / len(label),
                                                                100. * correct_or_not.sum().item() / len(label)))


if __name__ == '__main__':
    main()
