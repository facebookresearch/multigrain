# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.nn import functional as F


# Classification metrics

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k * (100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HistoryMeter(object):
    """Remember all values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.partials = []
        self.count = 0
        self.val = 0

    def update(self, x, n=1):
        self.val = x
        self.hist.append(x)
        x = n * x
        self.count += n
        # full precision summation based on http://code.activestate.com/recipes/393090/
        i = 0
        for y in self.partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                self.partials[i] = lo
                i += 1
            x = hi
        self.partials[i:] = [x]

    @property
    def avg(self):
        """
        Alternative to AverageMeter without floating point errors
        """
        return sum(self.partials, 0.0) / self.count if self.partials else 0


# Retrieval metrics

def score_ap(ranks, nres):
    """
    Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


def get_distance_matrix(outputs):
    """Get distance matrix given all embeddings."""
    square = torch.sum(outputs ** 2.0, dim=1, keepdim=True)
    distance_square = square + square.t() - (2.0 * torch.matmul(outputs, outputs.t()))
    return F.relu(distance_square) ** 0.5


def retrieval_acc(output, target, instances=4):
    """
    UKB-like accuracy criterion.
    Must be applied to the whole dataset.
    """
    _, pred = output.topk(instances, 1, True, False)
    d_mat -= torch.eye(d_mat.size(0))
    # d_mat_ic -= torch.eye(d_mat.size(0))
    _, pred = torch.sort(d_mat, dim=1)

    d_mats = [get_distance_matrix(f) for f in features]
    preds = [torch.sort(d.cpu(), dim=1)[1] for d in d_mats]