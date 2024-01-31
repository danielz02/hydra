import os
from argparse import Namespace

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier

from utils.model import create_model

MODEL_CONFIG = Namespace(
    configs='configs/configs.yml', result_dir='./trained_models',
    exp_mode='pretrain', arch='wrn_28_4', num_classes=10, layer_type='dense',
    init_type='kaiming_normal', snip_init=False, k=1.0, scaled_score_init=False, scaled_score_init_k=6,
    scale_rand_init=False, freeze_bn=False, source_net='', is_semisup=False, semisup_data='tinyimages',
    semisup_name=None, semisup_fraction=1.0, scores_init_type=None, dataset='CIFAR10', batch_size=256,
    test_batch_size=128, normalize=False, data_dir='./datasets', data_fraction=1.0, image_dim=32,
    mean=(0, 0, 0), std=(1, 1, 1), trainer='drt', epochs=150, optimizer='sgd', wd=0.0001, lr=0.1, lr_schedule='step',
    momentum=0.9, warmup_epochs=0, warmup_lr=0.1, save_dense=True, n_repeats=4, epsilon=0.031, num_steps=10,
    step_size=0.0078, clip_min=0, clip_max=1, distance='l_inf', const_init=False, beta=6.0, evaluate=False,
    val_method='smooth', start_epoch=0, gpu='0', no_cuda=False, seed=1234, print_freq=100, schedule_length=0,
    mixtraink=1, num_models=5, coeff=2.0, lamda=2.0, scale=5.0, plus_adv=False, adv_eps=0.2, init_eps=0.1,
    noise_std=0.25, lhs_weights=0.1, rhs_weights=0.5, lr_step=50, trades_loss=False, adv_training=False,
    drt_stab=False, separate_semisup=False, num_noise_vec=2, drt_consistency=True, lbd=20.0, attack='DDN',
    warmup=1, no_grad_attack=False, init_norm_DDN=256.0, gamma_DDN=0.05, layerwise=False, beta_div=None,
    sample_per_batch=False, subnet_to_subspace=False, subspace_type=None, ddp=False, gradient_predivide_factor=1.0,
    fp16_allreduce=False, amp=False
)


def get_care_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = DRTModel(model_kwargs, wrapper_kwargs, weights_path)
    model.eval()

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(3, 32, 32),
        channels_first=True,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model


class DRTModel(nn.Module):
    def __init__(self, model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None):
        super(DRTModel, self).__init__()

        self.noise_sd = model_kwargs.get('noise_sd', 0.0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Set path
        self.checkpoint = {}
        ckpt = torch.load(weights_path, map_location=self.device)["state_dict"]
        for k, v in ckpt.items():
            self.checkpoint[k.replace(".module", "")] = v
        dir_name, _ = os.path.split(weights_path)

        # Extract additional values from model_kwargs if needed
        # For example: some_value = model_kwargs.get('some_key', default_value)

        # Initialize base classifier
        self.base_classifier = create_model(MODEL_CONFIG, [], "cuda", None)
        self.base_classifier.load_state_dict(self.checkpoint)

    @torch.inference_mode()
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Number of noisy samples per input
        num_samples = 100

        # Get the shape of the input tensor
        N = x.size(0)
        input_shape = x.size()[1:]

        if self.noise_sd == 0.:
            mean_outputs = self.base_classifier(x)
        else:
            # List to store mean outputs
            mean_outputs = []

            # Process each x[i] tensor separately
            for i in range(N):
                # Generate Gaussian noise for x[i]
                noise = torch.randn(num_samples, *input_shape).to(x.device) * self.noise_sd

                # Add noise to x[i]
                noisy_samples = x[i] + noise

                # Pass noisy samples through the model
                outputs = self.base_classifier(noisy_samples).mean(dim=0)

                mean_outputs.append(outputs)

            # Convert the list of mean outputs to a tensor
            mean_outputs = torch.stack(mean_outputs)

        return self.confidences_to_log_softmax(mean_outputs)

    def confidences_to_log_softmax(self, confidences: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        # Clamp the confidences to avoid log(0) and log(1)
        confidences = torch.clamp(confidences, epsilon, 1 - epsilon)

        # Convert probabilities to logits
        logits = torch.log(confidences) - torch.log1p(-confidences)

        # Normalize logits for numerical stability
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values

        # Apply log_softmax
        log_softmax_values = F.log_softmax(logits, dim=-1)

        return log_softmax_values
