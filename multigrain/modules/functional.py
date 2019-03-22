# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import math
from torch.nn import functional as F


def add_bias_channel(x, dim=1):
    one_size = list(x.size())
    one_size[dim] = 1
    one = x.new_ones(one_size)
    return torch.cat((x, one), dim)


def flatten(x, keepdims=False):
    """
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y


def gem(x, p=3, eps=1e-6, clamp=True, add_bias=False, keepdims=False):
    if p == math.inf or p is 'inf':
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    if p == 1 and not (torch.is_tensor(p) and p.requires_grad):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    if clamp:
        x = x.clamp(min=eps)
    x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    if add_bias:
        x = add_bias_channel(x)
    if not keepdims:
        x = flatten(x)
    return x


def apply_pca(vt, pca_P=None, pca_m=None):
    B, C, H, W = vt.size()

    if pca_P is not None or pca_m is not None:
        vt = vt.permute(0, 2, 3, 1)

        vt = vt.view(B * H * W, 1, C)

        if pca_m is not None:
            vt = vt - pca_m.view(1, 1, -1)
        if pca_P is not None:
            vt = torch.matmul(vt, pca_P)

        vt = vt.view(B, H, W, C)
        vt = vt.permute(0, 3, 1, 2)
    return vt


def l2n(x, eps=1e-6, dim=1):
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x