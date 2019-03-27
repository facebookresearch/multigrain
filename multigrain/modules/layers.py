# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from torch.nn import functional as F
from . import functional as LF


class Select(nn.Module):
    """
    Apply module on subset of the input channels.
    Optionally drops other channels
    """

    def __init__(self, module, end, start=0, drop_other=False):
        super().__init__()
        self.module = module
        self.drop_other = drop_other
        self.end = end
        self.start = start

    def forward(self, x):
        bits = []
        bits.append(x[:, :self.start, ...])
        bits.append(x[:, self.start:self.end, ...])
        bits.append(x[:, self.end:, ...])
        bits[1] = self.module(bits[1])
        if self.drop_other:
            return bits[1]
        return torch.cat([b for b in bits if b.numel()], 1)


class Layer(nn.Module):
    """
    General module wrapper for a functional layer.
    """
    def __init__(self, name, **kwargs):
        super().__init__()
        self.name = name
        for n, v in kwargs.items():
            if torch.is_tensor(v):
                if v.requires_grad:
                    setattr(self, n, nn.Parameter(v))
                else:
                    self.register_buffer(n, v)
                kwargs[n] = 'self.' + n
        self.kwargs = kwargs

    def forward(self, input):
        kwargs = self.kwargs.copy()
        for (n, v) in kwargs.items():
            if isinstance(v, str) and v.startswith('self.'):
                kwargs[n] = getattr(self, v[len('self.'):])
        out = getattr(LF, self.name)(input, **kwargs)
        return out

    def __repr__(self):
        kwargs = []
        for (left, right) in self.kwargs.items():
            rt = repr(right)
            if isinstance(right, str) and right.startswith('self.'):
                vs = right[len('self.'):]
                v = getattr(self, vs)
                if vs in self._buffers and v.numel() <= 1:
                    rt = v
            kwargs.append('{}={}'.format(left, rt))
        kwargs = ', '.join(kwargs)
        if kwargs:
            kwargs = ', ' + kwargs
        return 'Layer(name=' + repr(self.name) + kwargs + ')'

