# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
from collections import OrderedDict as OD


class MultiCriterion(nn.Module):
    """
    Holds a dict of multiple losses with a weighting factor for each loss.
    - losses_dict: should be a dict with name as key and (loss, input_keys, weight) as values.
    - skip_zero: skip the computation of losses with 0 weight
    """
    def __init__(self, losses_dict, skip_zeros=False):
        super().__init__()
        self.losses = OD()
        self.input_keys = OD()
        self.weights = OD()
        for name, (loss, input_keys, weight) in losses_dict.items():
            self.losses[name] = loss
            self.input_keys[name] = input_keys
            self.weights[name] = weight
        self.losses = nn.ModuleDict(self.losses)
        self.skip_zeros = skip_zeros

    def forward(self, input_dict):
        return_dict = {}
        loss = 0.0
        for name, module in self.losses.items():
            for k in self.input_keys[name]:
                if k not in input_dict:
                    raise ValueError('Element {} not found in input.'.format(k))
            if self.weights[name] == 0.0 and self.skip_zeros:
                continue
            this_loss = module(*[input_dict[k] for k in self.input_keys[name]])
            return_dict[name] = this_loss
            loss = loss + self.weights[name] * this_loss
        return_dict['loss'] = loss
        return return_dict

