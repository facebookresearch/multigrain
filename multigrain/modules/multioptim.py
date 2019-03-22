# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict as OD


class MultiOptim(OD):
    """
    Holds dict of optimizers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = None

    def state_dict(self):
        D = OD()
        for k, v in self.items():
            for u, w in v.state_dict().items():
                D[k + '.' + u] = w
        return D

    def load_state_dict(self, D):
        for opt in self:
            local = {}
            for k, v in D.items():
                u, k2 = k.split('.', 1)
                if u == opt:
                    local[k2] = v
            self[opt].load_state_dict(local)
        return self

    def zero_grad(self):
        for opt in self.values():
            opt.zero_grad()

    def parameters(self):
        P = []
        for opt in self.values():
            for G in opt.param_groups:
                for p in G["params"]:
                    P.append(p)
        return P

    def step(self):
        for name, O in self.items():
            O.step()

    def set_base_lr(self):
        """
        Remember base learning rates to easily apply learning rate drops.
        """
        self.base_lr = {}
        for name, O in self.items():
            for i, G in enumerate(O.param_groups):
                self.base_lr[(name, i)] = G["lr"]

    def lr_multiply(self, multiplier):
        """
        Change lr multiplicatively relative to base_lr captured with self.set_base_lr().
        """
        for name, O in self.items():
            for i, G in enumerate(O.param_groups):
                G["lr"] = self.base_lr[(name, i)] * multiplier
