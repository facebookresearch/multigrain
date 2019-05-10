# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division, print_function, absolute_import
import torch
from torch import nn
from collections import OrderedDict as OD
from glob import glob
import os
import os.path as osp
from .logging import ordered_load, ordered_dump


class CheckpointHandler(object):
    """
    Save checkpoint and metric history in directory
    Remove old checkpoints, keep every save_every (none if 0)
    """

    def __init__(self, expdir, save_every=0, verbose=True, prefix='checkpoint_',
                 metrics_file='metrics'):
        self.expdir = expdir
        self.save_every = save_every
        self.verbose = verbose
        self.prefix = prefix
        self.metrics = metrics_file

    def available(self, dir=None):
        if dir is None:
            dir = self.expdir
        avail = OD()
        for checkpoint in glob(osp.join(dir, self.prefix + '*.pth')):
            epoch = int(osp.basename(checkpoint)[len(self.prefix):-len('.pth')])
            avail[epoch] = checkpoint
        return avail

    def exists(self, resume, dir=None):
        if dir is None:
            dir = self.expdir
        if resume in (-1, 0):
            return True
        avail = self.available(dir)
        return (resume in avail)

    def delete_old_checkpoints(self, epoch):
        avail = self.available()
        for k in avail:
            if k != epoch and (self.save_every == 0 or (k % self.save_every) != 0):
                os.remove(avail[k])

    def save_metrics(self, metrics_history):
        ordered_dump(metrics_history, osp.join(self.expdir, self.metrics + '.yaml'))

    def save(self, model, epoch, optimizer=None, metrics_history=None, extra=None):
        module = model.module if isinstance(model, nn.DataParallel) else model
        check = dict(model_state=module.state_dict())
        if optimizer is not None:
            check['optimizer_state'] = optimizer.state_dict()
        if extra is not None:
            check['extra'] = extra
        torch.save(check, osp.join(self.expdir, self.prefix + '{:d}.pth'.format(epoch)))
        if metrics_history is not None:
            self.save_metrics(metrics_history)
        self.delete_old_checkpoints(epoch)
        if self.verbose:
            print('Saved checkpoint in', self.expdir)

    def load_state_dict(self, model, state_dict):
        module = model.module if isinstance(model, nn.DataParallel) else model
        module.load_state_dict(state_dict)

    def resume(self, model, optimizer=None, metrics_history={}, resume_epoch=-1, resume_from=None, return_extra=True):
        """
        Restore model state dict and metrics.
        """
        if not resume_from:
            resume_from = self.expdir

        if osp.isdir(resume_from):
            avail = self.available(resume_from)
            if resume_epoch == -1:
                avail_keys = sorted(avail.keys())
                resume_epoch = avail_keys[-1] if avail_keys else 0

            if resume_epoch != 0:
                if resume_epoch not in avail:
                    raise ValueError('Epoch {} not found in {}'.format(resume_epoch, resume_from))
                resume_from = avail[resume_epoch]

        metrics_file = osp.join(self.expdir, self.metrics + '.yaml')

        if osp.isfile(metrics_file):
            metrics_history.clear()
            metrics_history.update(ordered_load(metrics_file))
        else:
            print('Reinitializing metrics, metrics file: {}'.format(metrics_file))

        if resume_epoch == 0:
            if self.verbose:
                print('Initialized model, optimizer')

            return 0, {} if return_extra else 0

        checkpoint = torch.load(resume_from, map_location=torch.device('cpu'))
        self.load_state_dict(model, checkpoint['model_state'])
        if self.verbose:
            print('Model state loaded from', resume_from)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                if self.verbose:
                    print('Optimizer state loaded from', resume_from)
            elif self.verbose:
                    print('No optimizer state found in', resume_from)

        if return_extra:
            extra = checkpoint.get('extra', {})
            return resume_epoch, extra
        return resume_epoch
