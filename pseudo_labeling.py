import os.path
import pickle
import torch
import logging
import argparse
from torchvision import transforms
from tqdm import tqdm

from utils.model import create_model
from utils.semisup import get_semisup_dataloader

parser = argparse.ArgumentParser(description="Label Tiny ImageNet data")
parser.add_argument(
    "--normalize",
    action="store_true",
    default=False,
    help="whether to normalize the data",
)
parser.add_argument("--data-dir", type=str, default="./datasets", help="path to datasets")
parser.add_argument("--outfile", type=str, help="name of the output file")
parser.add_argument(
    "--batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for training (default: 256)",
)
parser.add_argument(
    "--base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("--arch", type=str, default="vgg16_bn", help="Model architecture")
parser.add_argument(
    "--num-classes", type=int, default=10, help="Number of output classes in the model",
)
parser.add_argument(
    "--layer-type", type=str, choices=("dense", "subnet", "curve", "line"), help="dense | subnet layers"
)
parser.add_argument("--exp-mode", type=str, default="pretrain")
parser.add_argument(
    "--scores_init_type",
    choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
    help="Which init to use for relevance scores",
    default="kaiming_normal"
)
parser.add_argument(
    "--noise_std", type=float, default=0.25, help="noise hyperparameter"
)
parser.add_argument(
    "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
)
parser.add_argument("--num-models", type=int, default=3)
parser.add_argument(
    "--freeze-bn",
    action="store_true",
    default=False,
    help="freeze batch-norm parameters in pruning",
)
parser.add_argument("--k", type=float)
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.info(args)

    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}")

    checkpoint = torch.load(args.base_classifier, map_location=device)
    base_classifier = create_model(args, gpu_list, device, logger)
    base_classifier.load_state_dict(checkpoint["state_dict"])

    base_classifier.eval()

    args.semisup_data = "tinyimages"
    args.semisup_fraction = 1
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    sm_loader = get_semisup_dataloader(args, trans)

    margins = []
    pseudo_labels = []
    softmax = torch.nn.Softmax(1)
    with torch.no_grad():
        for img, _ in tqdm(sm_loader):
            y_pred = []
            margin = []
            img = img.to(device)
            for m in base_classifier.models:
                logits = m(img)
                prob, _ = softmax(logits).sort()
                y_pred.append(logits.argmax(axis=1).detach().cpu().numpy())
                margin.append((prob[:, -1] - prob[:, -2]).detach().cpu().numpy())

            margins.append(margin)
            pseudo_labels.append(y_pred)

    with open("datasets/tiny_images/ti_top_50000_pred_v3.1.pickle", "rb") as src:
        pkl = pickle.load(src)

    pkl["margins"] = margins
    pkl["extrapolated_targets"] = pseudo_labels
    pkl["prediction_model"] = args.base_classifier
    pkl["prediction_model_epoch"] = 150

    with open(os.path.join(args.data_dir, args.outfile), "wb") as f:
        pickle.dump(pkl, f)
