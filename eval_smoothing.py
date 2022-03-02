# evaluate a smoothed classifier on a dataset
import argparse
import sys
from time import time
import datetime
import importlib
import logging

import torch
import torch.nn as nn

import data

from utils.smoothing import Smooth
from utils.model import create_model

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument(
    "--dataset",
    type=str,
    choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet"),
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
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=128,
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
    "--layer-type", type=str, choices=("dense", "subnet", "curve"), help="dense | subnet layers"
)
parser.add_argument(
    "--noise_std", type=float, default=0.25, help="noise hyperparameter"
)
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")

# parser.add_argument(
#    "--split", choices=["train", "test"], default="test", help="train or test set"
# )
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

parser.add_argument("--k", type=float)
parser.add_argument("--sub-model", type=int)
parser.add_argument("--subspace-alpha", type=float)
parser.add_argument("--validation-only", action="store_true")

args = parser.parse_args()

if __name__ == "__main__":

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.outfile + ".log", "a"))
    logger.info(args)

    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}")

    # Create model
    if args.layer_type == "curve":
        args.layerwise = False
    base_classifier = create_model(args, gpu_list, device, logger)

    checkpoint = torch.load(args.base_classifier, map_location=device)
    base_classifier.load_state_dict(checkpoint["state_dict"])

    if args.sub_model is not None:
        base_classifier = base_classifier.models[args.sub_model]

    if args.layer_type == "curve":
        base_classifier.fixed_alpha = args.subspace_alpha

    base_classifier.eval()
    smoothed_classifier = Smooth(base_classifier, args.num_classes, args.noise_std)

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # Dataset
    D = data.__dict__[args.dataset](args)
    train_loader, test_loader = D.data_loaders()
    dataset = test_loader.dataset  # Certify test inputs only (default)

    val = getattr(importlib.import_module("utils.eval"), "smooth")
    p, r = val(base_classifier, device, test_loader, nn.CrossEntropyLoss(), args, None)
    logger.info(f"Validation natural accuracy for source-net: {p}, radisu: {r}")

    # for i, v in base_classifier.named_modules():
    #     if isinstance(v, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #         v.track_running_stats = False

    # eval_quick_smoothing(base_classifier, train_loader, device, sigma=0.25, nbatch=10)

    if args.validation_only:
        sys.exit()

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.to(device)
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch_size, device
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
