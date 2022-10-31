# coding: utf-8
import onnx
import torch
from utils.model import ckpt_combine
from utils.model import create_model
from argparse import Namespace
import models
from collections import OrderedDict
from utils.model import get_layers

# import horovod.torch as hvd
# hvd.init()

args = Namespace()
args.init_type = "kaiming_normal"
args.num_models = 3  # 3
args.layer_type = "Dense"  # "Subnet"
args.dataset = "cifar"  # "ImageNet"
args.normalize = False  # True
args.ddp = False
args.layer_type = "dense"
args.arch = "wrn_28_4"  # "ResNet50"
args.num_classes = 10  # 1000
args.k = 1.0
args.exp_mode = "pretrain"
args.freeze_bn = False
args.scaled_score_init = True
args.scores_init_type = None
args.gpu = "0"
model = create_model(args, [], "cuda", None)
ckpt_path = "trained_models/no-semisup-wrn_28_4-trainer_stab-3-beta_7-sigma_0.25-lr_0.1-0.1-0.01-step-k_0.1-cifar10-" \
            "epochs_150-40-40-batch_128/pretrain/latest_exp/checkpoint/checkpoint_dense.pth.tar"

new_state_dict = OrderedDict()
ckpt = torch.load(ckpt_path)
for k, v in ckpt["state_dict"].items():
    name = f"models.{k[7:]}".replace(".module", "")  # remove `module.`
    new_state_dict[name] = v
ckpt["state_dict"] = new_state_dict
# print(ckpt["state_dict"].keys())
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
# sample_input = torch.rand((1, 3, 224, 224)).cuda()
sample_input = torch.rand((1, 3, 32, 32)).cuda()
torch.onnx.export(
    model,
    sample_input,
    ckpt_path.replace(".pth.tar", ".onnx"),
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
onnx.load(ckpt_path.replace(".pth.tar", ".onnx"))
print(ckpt_path.replace(".pth.tar", ".onnx"))
