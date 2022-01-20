"""
A noisy convolution 2d for pytorch
Adapted from:
- https://raw.githubusercontent.com/Scitator/Run-Skeleton-Run/master/common/modules/NoisyLinear.py
- https://github.com/pytorch/pytorch/pull/2103/files#diff-531f4c06f42260d699f43dabdf741b6d
More details can be found in the paper `Noisy Networks for Exploration`
Original: https://gist.github.com/wassname/001aff274c7c8196055fabfc06cf80c5
"""
import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.autograd import Variable

from torch.nn.modules.utils import _single, _pair, _triple

class NoisyConv2d(Module):
    """Applies a noisy conv2d transformation to the incoming data:
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_channels: size of each input sample
        out_channels: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,
            defaults to 0.017 for independent and 0.4 for factorised. Default: None
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = NoisyConv2d(4, 2, (3,1))
        >>> input = torch.autograd.Variable(torch.randn(1, 4, 51, 3))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=1, dilation=1, groups=1, factorised=True, std_init=None):
        super(NoisyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.factorised = factorised
        
        self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size))
        self.weight_sigma = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = torch.Tensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon = None
            for dim in self.weight_sigma.size():
                if epsilon is None:
                    epsilon = self.scale_noise(dim)
                else:
                    epsilon = epsilon.unsqueeze(-1)*self.scale_noise(dim)
            weight_epsilon = Variable(epsilon)
            bias_epsilon = Variable(self.scale_noise(self.out_channels))
        else:
            weight_epsilon = Variable(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).normal_())
            bias_epsilon = Variable(torch.Tensor(self.out_channels).normal_())
        return F.conv2d(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon),
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups
                       )

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias_mu is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)