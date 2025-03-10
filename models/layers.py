import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def _get_weight(self):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj

        return self.w

    def forward(self, x):
        self.w = self._get_weight()

        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        # self.register_buffer('w', None)

    def set_prune_rate(self, k):
        self.k = k

    def _get_weight(self):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj

        return self.w

    def forward(self, x):
        self.w = self._get_weight()
        x = F.linear(x, self.w, self.bias)

        return x


StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d


class SubspaceConv(SubnetConv):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class SubspaceLinear(SubnetLinear):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(x, w, self.bias)
        return x


class SubspaceBN(nn.BatchNorm2d):
    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                    self.running_var is None
            )
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs["bias"] = False

        self.conv1 = SubnetConv(*args, **kwargs)

    @torch.no_grad()
    def initialize(self, initialize_fn):
        initialize_fn(self.conv1.weight)


class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs["bias"] = False
        self.linear1 = SubnetLinear(*args, **kwargs)

    @torch.no_grad()
    def initialize(self, initialize_fn):
        initialize_fn(self.linear1.weight)


class ThreeParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs["bias"] = False

        self.conv1 = SubnetConv(*args, **kwargs)
        self.conv2 = SubnetConv(*args, **kwargs)

    @torch.no_grad()
    def initialize(self, initialize_fn):
        initialize_fn(self.conv1.weight)
        initialize_fn(self.conv2.weight)


class TwoParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)


class ThreeParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)


class LinesConv(TwoParamConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        w = (1 - self.alpha) * self._get_weight() + self.alpha * self.conv1._get_weight()
        return w


class LinesLinear(TwoParamLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        w = (1 - self.alpha) * self._get_weight() + self.alpha * self.linear1._get_weight()
        return w


class LinesBN(TwoParamBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


class CurvesConv(ThreeParamConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        if getattr(self, "w0", False):
            w = self._get_weight()
        elif getattr(self, "w1", False):
            w = self.conv1._get_weight()
        elif getattr(self, "w2", False):
            w = self.conv2._get_weight()
        else:
            w = (
                    ((1 - self.alpha) ** 2) * self._get_weight()
                    + 2 * self.alpha * (1 - self.alpha) * self.conv2._get_weight()
                    + (self.alpha ** 2) * self.conv1._get_weight()
            )
        return w


class CurvesLinear(CurvesConv):
    def __init__(self, in_features, out_features, *args, **kwargs):
        kwargs["kernel_size"] = 1
        kwargs["in_channels"] = in_features
        kwargs["out_channels"] = out_features

        super().__init__(*args, **kwargs)


class CurvesBN(ThreeParamBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weight(self):
        if getattr(self, "w0", False):
            w = self.weight, b = self.bias
        elif getattr(self, "w1", False):
            w = self.weight1, b = self.bias1
        elif getattr(self, "w2", False):
            w = self.weight2, b = self.bias2
        else:
            w = (
                ((1 - self.alpha) ** 2) * self.weight
                + 2 * self.alpha * (1 - self.alpha) * self.weight2
                + (self.alpha ** 2) * self.weight1
            )
            b = (
                ((1 - self.alpha) ** 2) * self.bias
                + 2 * self.alpha * (1 - self.alpha) * self.bias2
                + (self.alpha ** 2) * self.bias1
            )
        return w, b


if __name__ == "__main__":
    m = LinesLinear(in_features=3, out_features=3)
    print(isinstance(m, nn.Conv2d))
