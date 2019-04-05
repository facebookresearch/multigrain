# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Monkey-patches https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/pnasnet.py
The source implementation cannot be applied to images of arbitrary shape
Here we add cropping operations before additions and concatenations to address this.
"""
import pretrainedmodels

import torch


def equal_except(a, b, avoid=None):
    for i, (ai, bi) in enumerate(zip(a, b)):
        if ai != bi and (avoid is None or i != avoid):
            return False
    return True


def shrink_common(*tensors, avoid=None):
    sizes = [tuple(t.size()) for t in tensors]
    st = tuple(min(*dims) for dims in zip(*sizes))
    out_tensors = []
    for t, s in zip(tensors, sizes):
        if not equal_except(s, st, avoid):
            dest_size = list(st)
            if avoid is not None:
                dest_size[avoid] = s[avoid]
            t = t.__getitem__(list(slice(si) for si in dest_size))
        out_tensors.append(t)
    return out_tensors


def shrink_sum(*tensors):
    tensors = shrink_common(*tensors)
    return sum(tensors)

def shrink_cat(tensors, dim=1):
    tensors = shrink_common(*tensors, avoid=dim)
    return torch.cat(tensors, dim=1)


def cell_forward(self, x_left, x_right):
    x_comb_iter_0_left = self.comb_iter_0_left(x_left)
    x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

    x_comb_iter_1_left = self.comb_iter_1_left(x_right)
    x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

    x_comb_iter_2_left = self.comb_iter_2_left(x_right)
    x_comb_iter_2_right = self.comb_iter_2_right(x_right)
    x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_comb_iter_2_right)

    x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
    x_comb_iter_3_right = self.comb_iter_3_right(x_right)
    x_comb_iter_3 = shrink_sum(x_comb_iter_3_left, x_comb_iter_3_right)

    x_comb_iter_4_left = self.comb_iter_4_left(x_left)
    if self.comb_iter_4_right:
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
    else:
        x_comb_iter_4_right = x_right
    x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_comb_iter_4_right)

    x_out = shrink_cat(
        [x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3,
         x_comb_iter_4], 1)
    return x_out


def pnasnet5large(*args, num_classes=1000, **kwargs):
    pretrainedmodels.models.pnasnet.CellBase.cell_forward = cell_forward
    model = pretrainedmodels.models.pnasnet5large(*args, num_classes=num_classes, **kwargs)
    return model
