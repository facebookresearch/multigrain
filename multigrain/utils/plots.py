# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os.path as osp


def make_plots(metrics_history, destdir):
    keys = set()
    for metrics in metrics_history.values():
        keys.update(metrics.keys())

    groups = defaultdict(list)
    for k in keys:
        split = k.split('_', 1)
        if len(split) == 1:
            split = [''] + split
        subk, g = split
        groups[g].append((subk, k))

    for g in groups:
        plt.figure()
        plt.title(g)
        for k, kg in groups[g]:
            epochs = []
            values = []
            for epoch, metrics in metrics_history.items():
                if kg in metrics:
                    if isinstance(metrics[kg], list):
                        for i, v in enumerate(metrics[kg]):
                            epochs.append(epoch - 1 + (i + 1)/len(metrics[kg]))
                            values.append(v)
                    else:
                        epochs.append(epoch)
                        values.append(metrics[kg])
            plt.plot(epochs, values, 'o-', label=k if k else None)
        if len(groups[g]) > 1:
            plt.legend()
        plt.xlabel("epochs")
        plt.tight_layout()
        plt.savefig(osp.join(destdir, g + '.pdf'))
