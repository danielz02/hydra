# coding: utf-8
import torch
from argparse import Namespace
from torch.nn.utils import prune
from collections import OrderedDict
from utils.model import create_model
from torch.nn.utils.prune import ln_structured


ckpt_path = "trained_models/no-semisup-wrn_28_4-trainer_stab-3-beta_7-sigma_0.25-lr_0.1-0.1-0.01-step-" \
            "k_0.1-cifar10-epochs_150-40-40-batch_128/prune/latest_exp/checkpoint/model_best.pth.tar"

args = Namespace()
args.init_type = "kaiming_normal"
args.num_models = 3
args.dataset = "cifar"
args.normalize = False
args.ddp = False
args.layer_type = "subnet"
args.arch = "wrn_28_4"
args.num_classes = 10
args.k = 0.1
args.exp_mode = "prune"
args.freeze_bn = False
args.scaled_score_init = True
args.scores_init_type = None
args.gpu = "0"
model = create_model(args, [], "cuda", None)
new_state_dict = OrderedDict()
ckpt = torch.load(ckpt_path)
for k, v in ckpt["state_dict"].items():
    name = f"models.{k[7:]}".replace(".module", "")
    new_state_dict[name] = v
ckpt["state_dict"] = new_state_dict
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

for k, v in model.named_modules():
    if "conv" in k and "models.0.conv1" not in k and "models.1.conv1" not in k and "models.2.conv1" not in k and \
            "convShortcut" not in k and "block1.layer.0.conv1" not in k:
        ln_structured(v, "weight", amount=(1 - args.k), n=1, dim=1, importance_scores=v.popup_scores)
        with torch.no_grad():
            v.popup_scores *= v.weight_mask
        prune.remove(v, 'weight')

for k, v in model.named_parameters():
    if "conv" in k:
        print(k, (torch.count_nonzero(v) / v.numel()).item())

ckpt["state_dict"] = model.state_dict()
torch.save(ckpt, ckpt_path.replace(".pth.tar", ".structured.pth.tar"))
print(ckpt_path.replace(".pth.tar", ".structured.pth.tar"))
