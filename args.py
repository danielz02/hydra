import argparse


# Inherited from https://github.com/yaodongyu/TRADES/blob/master/train_trades_cifar10.py
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default="", help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./trained_models",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    parser.add_argument(
        "--exp-mode",
        type=str,
        choices=("pretrain", "prune", "finetune"),
        help="Train networks following one of these methods.",
    )

    # Model
    parser.add_argument("--arch", type=str, help="Model achitecture")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in the model",
    )
    parser.add_argument(
        "--layer-type", type=str, choices=("dense", "subnet", "curve", "line"), help="dense | subnet"
    )
    parser.add_argument(
        "--init_type",
        choices=("kaiming_normal", "kaiming_uniform", "signed_const"),
        help="Which init to use for weight parameters: kaiming_normal | kaiming_uniform | signed_const",
    )

    # Pruning
    parser.add_argument(
        "--snip-init",
        action="store_true",
        default=False,
        help="Whether implemnet snip init",
    )

    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Fraction of weight variables kept in subnet",
    )

    parser.add_argument(
        "--scaled-score-init",
        action="store_true",
        default=False,
        help="Init importance scores proportional to weights (default kaiming init)",
    )

    parser.add_argument(
        "--scaled-score-init-k",
        type=int,
        default=6,
        help="Init importance scores proportional to weights (default kaiming init)",
    )

    parser.add_argument(
        "--scale_rand_init",
        action="store_true",
        default=False,
        help="Init weight with scaling using pruning ratio",
    )

    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        default=False,
        help="freeze batch-norm parameters in pruning",
    )

    parser.add_argument(
        "--source-net",
        type=str,
        default="",
        help="Checkpoint which will be pruned/fine-tuned",
    )

    # Semi-supervision dataset setting
    parser.add_argument(
        "--is-semisup",
        action="store_true",
        default=False,
        help="Use semisupervised training",
    )

    parser.add_argument(
        "--semisup-data",
        type=str,
        choices=("tinyimages", "splitgan"),
        help="Name for semi-supervision dataset",
    )

    parser.add_argument(
        "--semisup-name",
        type=str,
        help="Path for semi-supervision dataset",
    )

    parser.add_argument(
        "--semisup-fraction",
        type=float,
        default=0.25,
        help="Fraction of images used in training from semisup dataset",
    )

    # Randomized smoothing
    # parser.add_argument(
    #     "--noise-std",
    #     type=float,
    #     default=0.25,
    #     help="Std of normal distribution used to generate noise",
    # )

    # parser.add_argument(
    #     "--scale_rand_init",
    #     action="store_true",
    #     default=False,
    #     help="Init weight with scaling using pruning ratio",
    # )

    parser.add_argument(
        "--scores_init_type",
        choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
        help="Which init to use for relevance scores",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet", "TinyImageNet"),
        help="Dataset for training and eval",
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
        "--normalize",
        action="store_true",
        default=False,
        help="whether to normalize the data",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./datasets", help="path to datasets"
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        help="Fraction of images used from training set",
    )
    parser.add_argument(
        "--image-dim", type=int, default=32, help="Image size: dim x dim x 3"
    )
    parser.add_argument(
        "--mean", type=tuple, default=(0, 0, 0), help="Mean for data normalization"
    )
    parser.add_argument(
        "--std", type=tuple, default=(1, 1, 1), help="Std for data normalization"
    )

    # Training
    parser.add_argument(
        "--trainer",
        type=str,
        default="base",
        choices=("base", "adv", "mixtrain", "crown-ibp", "smooth", "freeadv", "trs", "drt", "gaussian", "consistency",
                 "smoothadv"),
        help="Natural (base) or adversarial or verifiable training",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )
    parser.add_argument("--wd", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("step", "cosine"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--warmup-epochs", type=int, default=-1, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=0.1, help="warmup learning rate"
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        default=False,
        help="Save dense model alongwith subnets.",
    )

    # Free-adv training (only for imagenet)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=4,
        help="--number of repeats in free-adv training",
    )

    # Adversarial attacks
    parser.add_argument("--epsilon", default=64 / 255, type=float, help="perturbation")
    parser.add_argument(
        "--num-steps", default=10, type=int, help="perturb number of steps"
    )
    parser.add_argument(
        "--step-size", default=2.0 / 255, type=float, help="perturb step size"
    )
    parser.add_argument("--clip-min", default=0, type=float, help="perturb step size")
    parser.add_argument("--clip-max", default=1.0, type=float, help="perturb step size")
    parser.add_argument(
        "--distance",
        type=str,
        default="l_inf",
        choices=("l_inf", "l_2"),
        help="attack distance metric",
    )
    parser.add_argument(
        "--const-init",
        action="store_true",
        default=False,
        help="use random initialization of epsilon for attacks",
    )
    parser.add_argument(
        "--beta",
        default=6.0,
        type=float,
        help="regularization, i.e., 1/lambda in TRADES",
    )

    # Evaluate
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate model"
    )

    parser.add_argument(
        "--val_method",
        type=str,
        default="base",
        choices=("base", "adv", "mixtrain", "ibp", "smooth", "freeadv"),
        help="base: evaluation on unmodified inputs | adv: evaluate on adversarial inputs",
    )

    # Restart
    parser.add_argument(
        "--start-epoch",
        type=int,
        # default=0,
        help="manual start epoch (useful in restarts)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to latest checkpoint (default:None)",
    )

    # Additional
    parser.add_argument(
        "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="Number of batches to wait before printing training logs",
    )

    parser.add_argument(
        "--schedule_length",
        type=int,
        default=0,
        help="Number of epochs to schedule the training epsilon.",
    )

    parser.add_argument(
        "--mixtraink",
        type=int,
        default=1,
        help="Number of samples out of a batch to train with sym in mixtrain.",
    )

    parser.add_argument(
        "--num-models",
        type=int,
        default=3,
        help="Number of base models in the ensemble"
    )

    # TRS Arguments
    parser.add_argument('--coeff', default=2.0, type=float)
    parser.add_argument('--lamda', default=2.0, type=float)
    parser.add_argument('--scale', default=5.0, type=float)
    parser.add_argument('--plus-adv', action='store_true')
    parser.add_argument('--adv-eps', default=0.2, type=float)
    parser.add_argument('--init-eps', default=0.1, type=float)

    # DRT Arguments
    parser.add_argument('--noise-std', type=float)
    parser.add_argument('--lhs-weights', type=float)
    parser.add_argument('--rhs-weights', type=float)
    parser.add_argument('--lr-step', type=int, default=50)
    parser.add_argument('--trades-loss', action='store_true')
    parser.add_argument('--adv-training', action='store_true')
    parser.add_argument('--drt-stab', action='store_true', help="add STAB to DRT")
    parser.add_argument('--separate-semisup', action='store_true', help="Apply DRT to labeled data only")
    parser.add_argument('--num-noise-vec', default=2, type=int, help="number of noise vectors. `m` in the paper.")

    # Consistency Loss
    parser.add_argument('--drt-consistency', action='store_true', help="add Consistency Loss to DRT")
    parser.add_argument('--lbd', type=float, help="weight on Consistency Loss")

    # SmoothAdv Arguments
    parser.add_argument('--attack', default='DDN', type=str, choices=['DDN', 'PGD'])
    parser.add_argument(
        '--warmup', default=1, type=int, help="Number of epochs over which the maximum allowed perturbation increases"
                                              "linearly from zero to args.epsilon."
    )
    parser.add_argument(
        '--no-grad-attack', action='store_true', help="Choice of whether to use gradients during attack or do"
                                                      "the cheap trick"
    )

    # DDN-specific
    parser.add_argument('--init-norm-DDN', default=256.0, type=float)
    parser.add_argument('--gamma-DDN', default=0.05, type=float)

    # Self-ensemble
    parser.add_argument("--layerwise", action="store_true", help="Flag for enable layer-wise subspace sampling")
    parser.add_argument("--beta-div", type=float, help="Weight of cosine diversity regularization term")
    parser.add_argument("--sample-per-batch", action="store_true")
    parser.add_argument("--subnet-to-subspace", action="store_true")
    parser.add_argument("--subspace-type", type=str, help="Type of subspace to convert from")

    # DDP
    parser.add_argument("--ddp", action="store_true", help="Enable distributed training")
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument("--fp16-allreduce", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--amp", action="store_true", help="Enable automated mixed precision training")

    args = parser.parse_args()
    args.epsilon = args.epsilon / 255 if args.epsilon > 1 else args.epsilon

    if args.layerwise:
        assert args.layer_type in ["curve", "line"]
    if args.drt_consistency:
        assert args.lbd
    if args.separate_semisup:
        assert args.is_semisup

    assert not (args.drt_consistency and args.drt_stab)

    return args
