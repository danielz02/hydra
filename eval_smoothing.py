# evaluate a smoothed classifier on a dataset
import argparse
import os
import sys
from itertools import product
from time import time
import datetime
import importlib
import logging

import matplotlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from advertorch.attacks import L2PGDAttack
from advertorch.attacks.utils import attack_whole_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from ptflops import get_model_complexity_info

import data
from models.ensemble import SubspaceEnsemble, Subspace

from utils.smoothing import Smooth
from utils.model import create_model, subspace_to_subnet, dense_to_sparse
from utils.utils_ensemble import subspace_diversity, update_bn

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument(
    "--dataset",
    type=str,
    choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet", "TinyImageNet"),
    help="Dataset for training and eval",
)
parser.add_argument(
    "--normalize",
    action="store_true",
    default=False,
    help="whether to normalize the data",
)
parser.add_argument(
    "--data-dir", type=str, default="./datasets", help="path to datasets"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
parser.add_argument(
    "--data-fraction",
    type=float,
    default=1.0,
    help="Fraction of images used from training set",
)

parser.add_argument(
    "--base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("--arch", type=str, default="vgg16_bn", help="Model achitecture")
parser.add_argument(
    "--num-classes", type=int, default=10, help="Number of output classes in the model",
)
parser.add_argument(
    "--layer-type", type=str, choices=("dense", "subnet", "curve", "line"), help="dense | subnet layers"
)
parser.add_argument(
    "--noise-std", type=float, default=0.25, help="noise hyperparameter"
)
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument(
    "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument(
    "--print-freq",
    type=int,
    default=100,
    help="Number of batches to wait before printing training logs",
)
parser.add_argument("--num-models", type=int, default=3)
parser.add_argument("--exp-mode", type=str, default="pretrain")
parser.add_argument(
    "--freeze-bn",
    action="store_true",
    default=False,
    help="freeze batch-norm parameters in pruning",
)
parser.add_argument(
    "--scores_init_type",
    choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
    help="Which init to use for relevance scores",
    default="kaiming_normal"
)
parser.add_argument("--copy", type=int)
parser.add_argument("--k", type=float)
parser.add_argument("--sub-model", type=int)
parser.add_argument("--subspace-alpha", type=str)
parser.add_argument("--subspace-type", type=str)
parser.add_argument("--random-alpha", action="store_true")
parser.add_argument("--validation-only", action="store_true")
parser.add_argument("--subspace-accuracy", action="store_true")
parser.add_argument("--subspace-diversity", action="store_true")
parser.add_argument("--subnet-to-subspace", action="store_true")
parser.add_argument("--ddp", action="store_true")
parser.add_argument("--eval-trans", action="store_true")
parser.add_argument('--adv-eps', default=0.2, type=float)
parser.add_argument("--to-sparse", action="store_true")


def gen_plot(transmat):
    import itertools
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 3, step=1))
    plt.xticks(np.arange(0, 3, step=1))
    cmp = plt.get_cmap('Blues')
    plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
    plt.title("Transfer attack success rate")
    plt.colorbar()
    thresh = 50.0
    for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
        plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
                 horizontalalignment="center",
                 color="white" if transmat[i, j] > thresh else "black")

    plt.ylabel('Target model')
    plt.xlabel('Base model')
    outpath = "./"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    plt.savefig(outpath + "%s.pdf" % args.outfile, bbox_inches='tight')


def trans_matrix():
    trans = np.zeros((3, 3))
    models = base_classifier.models

    adv = []
    loss_fn = nn.CrossEntropyLoss()

    for i in range(len(models)):
        curmodel = models[i]  # Smooth(models[i], args.num_classes, args.noise_std)
        from utils.smoothadv import SmoothAdv_PGD
        adversary = SmoothAdv_PGD(model=curmodel, steps=20, max_norm=110 / 255)
        adv.append(adversary)

    for i in range(len(models)):
        test_iter = tqdm(test_loader, desc='Batch', leave=False, position=2)
        _, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device=device)
        for j in range(len(models)):
            i_success_count = 0
            for r in range((_.size(0) - 1) // 200 + 1):
                inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
                y = label[r * 200: min((r + 1) * 200, _.size(0))]
                smoothed_model = Smooth(models[j], args.num_classes, args.noise_std)
                adv_pred = []
                for x_adv in tqdm(inputc):
                    adv_pred.append(smoothed_model.predict(x_adv, args.N0, args.alpha, args.batch_size, device))
                # __ = adv[j].predict(inputc + torch.randn_like(inputc) * args.noise_std)
                # output = __.max(1, keepdim=False)[1]
                advpred_ = advpred[r * 200: min((r + 1) * 200, _.size(0))]
                output = torch.tensor(adv_pred, device=device).reshape(-1)
                i_success_count += (advpred_ != y).sum().item()
                if i != j:
                    trans[i, j] += torch.logical_and(output != y, advpred_ != y).sum().item()
                else:
                    trans[i, j] += (output != y).sum().item()
            if i == j:
                trans[i][j] /= len(label)
            else:
                trans[i][j] /= i_success_count
            print(f"{i} -> {j} Attack Successful Rate: {trans[i, j]}")

    print(trans * 100.)
    gen_plot(trans)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.ddp:
        import horovod.torch as hvd
        import torch.backends.cudnn as cudnn

        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        torch.set_num_threads(8)
        cudnn.benchmark = True
        is_rank0 = (args.ddp and hvd.rank() == 0) or not args.ddp

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.outfile + ".log", "a"))
    logger.info(args)

    if args.ddp:
        device = torch.device("cuda")
        gpu_list = ['cuda']
    else:
        gpu_list = [int(i) for i in args.gpu.strip().split(",")]
        device = torch.device(f"cuda:{gpu_list[0]}")

    # Create model
    if args.layer_type in ["curve", "line"]:
        args.layerwise = False
    base_classifier = create_model(args, gpu_list, device, logger)

    if args.subspace_alpha:
        args.subspace_alpha = [float(x.rstrip()) for x in args.subspace_alpha.split(",")]
        logger.info(f"Subspace Alpha: {args.subspace_alpha}")

    checkpoint = torch.load(args.base_classifier, map_location=device)
    if args.subnet_to_subspace:
        subspace_to_subnet(base_classifier, checkpoint["state_dict"], args.subspace_alpha, args.subspace_type)
    else:
        base_classifier.load_state_dict(checkpoint["state_dict"])
    if args.to_sparse:
        base_classifier = dense_to_sparse(base_classifier)
        print("Converted weights to sparse tensors!")
    if args.copy:
        assert args.num_models == 1
        base_classifier.models.extend([base_classifier.models[0]] * (args.copy - 1))
        print(f"Made {len(base_classifier.models)} copies of base classifier!")

    if args.sub_model is not None:
        base_classifier = base_classifier.models[args.sub_model]

    if args.layer_type in ["curve", "line"] and len(args.subspace_alpha) == 1:
        base_classifier.fixed_alpha = args.subspace_alpha[0]
    elif args.layer_type in ["curve", "line"] and len(args.subspace_alpha) > 1:
        base_classifier = SubspaceEnsemble(base_classifier, num_models=len(args.subspace_alpha))
        base_classifier.alpha = args.subspace_alpha

    base_classifier.eval()

    with torch.cuda.device(gpu_list[0]):
        macs, params = get_model_complexity_info(base_classifier, (3, 32, 32), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # Dataset
    D = data.__dict__[args.dataset](args)
    train_loader, test_loader = D.data_loaders()
    dataset = test_loader.dataset  # Certify test inputs only (default)

    val = getattr(importlib.import_module("utils.eval"), "smooth")
    p, r = val(base_classifier, device, test_loader, nn.CrossEntropyLoss(), args, None)
    logger.info(f"Validation natural accuracy for source-net: {p}, radius: {r}")

    if args.eval_trans:
        trans_matrix()

    if args.subspace_accuracy and len(args.subspace_alpha) == 0:
        accuracies = []
        for alpha in np.arange(0, 1.01, 0.05):
            base_classifier.fixed_alpha = alpha
            p, r = val(base_classifier, device, test_loader, nn.CrossEntropyLoss(), args, None)
            logger.info(f"Validation natural accuracy for source-net: {p}, radius: {r} at alpha = {alpha}")
            accuracies.append(p.cpu().item())
        df = pd.DataFrame({"Alpha": np.arange(0, 1.01, 0.05), "Natural Accuracy": accuracies})
        df.to_csv(f"{args.outfile}.csv", index=False)
    elif args.subspace_accuracy and isinstance(base_classifier, SubspaceEnsemble):
        record_rows = []
        for subspace_alphas in tqdm(product(np.arange(0, 1.01, 0.2), np.arange(0, 1.01, 0.2), np.arange(0, 1.01, 0.2))):
            base_classifier.alpha = subspace_alphas
            p, r = val(base_classifier, device, test_loader, nn.CrossEntropyLoss(), args, None)
            logger.info(f"Validation natural accuracy for source-net: {p}, radius: {r} at alpha = {subspace_alphas}")
            record_rows.append((*subspace_alphas, p.cpu().item()))
        df = pd.DataFrame(
            record_rows, columns=[f"Alpha {i}" for i in range(len(record_rows[0]) - 1)] + ["Benign Accuracy"]
        )
        df.to_csv(f"{args.outfile}.csv", index=False)

    if args.subspace_diversity and isinstance(base_classifier, Subspace):
        record_rows = []
        for subspace_alphas in tqdm(product(np.arange(0, 1.01, 0.2), np.arange(0, 1.01, 0.2), np.arange(0, 1.01, 0.2))):
            grad_loss = subspace_diversity(base_classifier, test_loader, subspace_alphas, device, args)
            logger.info(f"Subspace diversity: {grad_loss}, at alpha = {subspace_alphas}")
            record_rows.append((*subspace_alphas, grad_loss.cpu().item()))
        df = pd.DataFrame(
            record_rows, columns=[f"Alpha {i}" for i in range(len(record_rows[0]) - 1)] + ["Diversity Loss"]
        )
        df.to_csv(f"{args.outfile}.csv", index=False)

    # for i, v in base_classifier.named_modules():
    #     if isinstance(v, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #         v.track_running_stats = False

    # eval_quick_smoothing(base_classifier, train_loader, device, sigma=0.25, nbatch=10)

    if args.validation_only:
        sys.exit()

    smoothed_classifier = Smooth(base_classifier, args.num_classes, args.noise_std)

    for i in range(len(dataset)):
        if args.random_alpha and isinstance(smoothed_classifier.base_classifier, SubspaceEnsemble):
            delattr(smoothed_classifier.base_classifier, "fixed_alpha")
            smoothed_classifier.base_classifier.reset_alpha()
        elif args.random_alpha and isinstance(smoothed_classifier.base_classifier, Subspace):
            delattr(smoothed_classifier.base_classifier, "fixed_alpha")
            smoothed_classifier.base_classifier.fixed_alpha = np.random.uniform()

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        before_time = time()
        # certify the prediction of g around x
        x = x.to(device)
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.test_batch_size, device
        )
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed
            ),
            file=f,
            flush=True,
        )

    f.close()
