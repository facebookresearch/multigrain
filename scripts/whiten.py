# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import DataLoader, Subset

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import ListDataset
from multigrain import utils
from multigrain.backbones import backbone_list
from multigrain.lib.whiten import get_whiten

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os.path as osp
tic, toc = utils.Tictoc()


def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)

    argfile = osp.join(osp.join(args.expdir), 'whiten_args.yaml')

    args.cuda = not args.no_cuda

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    collate_fn = dict(collate_fn=list_collate) if args.input_crop == 'rect' else {}

    transforms = get_transforms(input_size=args.input_size, crop=(args.input_crop == 'square'), need=('val',), backbone=args.backbone)
    dataset = ListDataset(args.whiten_path, args.whiten_list, transforms['val'])
    if args.num_whiten_images != -1:
        dataset = Subset(dataset, list(range(args.num_whiten_images)))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, **collate_fn)

    model = get_multigrain(args.backbone, include_sampling=False, pretrained_backbone=args.pretrained_backbone)

    if args.cuda:
        model = utils.cuda(model)

    p = model.pool.p

    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        resume_epoch = checkpoints.resume(model, resume_epoch=args.resume_epoch,
                                          resume_from=args.resume_from, return_extra=False)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    if args.pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    print(model)

    model.init_whitening()
    model.eval()

    print("Computing embeddings...")
    embeddings = []
    for i, batch in enumerate(loader):
        if i % (len(loader) / 100) < 1:
            print("{}/{} ({}%)".format(i, len(loader), int(i // (len(loader) / 100))))
        with torch.no_grad():
            if args.cuda:
                batch = utils.cuda(batch)
            embeddings.append(model(batch)['embedding'].cpu())
    embeddings = torch.cat(embeddings)
    if args.no_include_last:
        embeddings = embeddings[:, :-1]

    print("Computing whitening...")
    m, P = get_whiten(embeddings)

    if args.no_include_last:
        # add an preserved channel to the PCA
        m = torch.cat((m, torch.tensor([0.0])), 0)
        D = P.size(0)
        P = torch.cat((P, torch.zeros(1, D)), 0)
        P = torch.cat((P, torch.cat((torch.zeros(D, 1), torch.tensor([1.0])), 1)), 1)

    model.integrate_whitening(m, P)

    if not args.dry:
        checkpoints.save(model, resume_epoch if resume_epoch != -1 else 0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Whitening computation for MultiGrain model, computes the whitening matrix",
                                         formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500_whitened', help='destination directory for checkpoint')
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    parser.add_argument('--resume-from', default=None, help='source experiment to whiten')
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    parser.add_argument('--pretrained-backbone', action='store_true', help='use pretrained backbone')
    parser.add_argument('--pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    parser.add_argument('--no-include-last', action='store_true', help='remove last channel from PCA (useful to not include "bias multiplier" channel)')
    parser.add_argument('--whiten-list', default='data/whiten.txt', help='list of images to compute whitening')
    parser.add_argument('--whiten-path', default='data/whiten', help='whitening data root')
    parser.add_argument('--num-whiten-images', default=-1, type=int, help='number of images used in whitening. (-1 -> all in list)')
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()

    run(args)
