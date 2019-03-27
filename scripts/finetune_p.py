# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import IN1K, IdDataset
from multigrain import utils
from multigrain.modules import MultiOptim
from multigrain.backbones import backbone_list

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os.path as osp
from collections import defaultdict, Counter
from collections import OrderedDict as OD
tic, toc = utils.Tictoc()


def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)
    argfile = osp.join(osp.join(args.expdir), 'finetune_p_args.yaml')

    if osp.isfile(argfile):
        oldargs = yaml.load(open(argfile))
        if oldargs != args.__dict__:
            print('WARNING: Changed configuration keys compared to stored experiment')
            utils.arguments.compare_dicts(oldargs, args.__dict__, verbose=True)

    args.cuda = not args.no_cuda
    args.validate_first = not args.no_validate_first
    args.validate = not args.no_validate

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    transforms = get_transforms(IN1K, args.input_size, crop=(args.input_crop == 'square'), need=('val',))
    datas = {}
    for split in ('train', 'val'):
        datas[split] = IdDataset(IN1K(args.imagenet_path, split, transform=transforms['val']))
    loaders = {}
    collate_fn = dict(collate_fn=list_collate) if args.input_crop == 'rect' else {}
    selected = []
    count = Counter()
    for i, label in enumerate(datas['train'].dataset.labels):
        if count[label] < args.images_per_class:
            selected.append(i)
            count[label] += 1
    datas['train'].dataset = Subset(datas['train'].dataset, selected)
    loaders['train'] = DataLoader(datas['train'], batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True, **collate_fn)
    loaders['val'] = DataLoader(datas['val'], batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, **collate_fn)

    model = get_multigrain(args.backbone, include_sampling=False,
                           pretrained_backbone=args.pretrained_backbone, learn_p=True)

    criterion = torch.nn.CrossEntropyLoss()
    if args.cuda:
        criterion = utils.cuda(criterion)
        model = utils.cuda(model)

    optimizers = OD()
    p = model.pool.p
    optimizers['p'] = SGD([p], lr=args.learning_rate, momentum=args.momentum)
    optimizers = MultiOptim(optimizers)

    def training_step(batch):
        optimizers.zero_grad()

        output_dict = model(batch['input'])
        loss = criterion(output_dict['classifier_output'], batch['classifier_target'])
        top1, top5 = utils.accuracy(output_dict['classifier_output'].data, batch['classifier_target'].data, topk=(1, 5))

        p.grad = torch.autograd.grad(loss, p)[0]  # partial backward
        optimizers.step()

        return OD([
            ('cross_entropy', loss.item()),
            ('p', p.item()),
            ('top1', top1),
            ('top5', top5),
        ])

    def validation_step(batch):
        with torch.no_grad():
            output_dict = model(batch['input'])
            target = batch['classifier_target']
            xloss = criterion(output_dict['classifier_output'], target)
            top1, top5 = utils.accuracy(output_dict['classifier_output'], target, topk=(1, 5))

        return OD([
            ('cross_entropy', xloss.item()),
            ('top1', top1),
            ('top5', top5),
        ])

    metrics_history = OD()

    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        checkpoints.resume(model, metrics_history=metrics_history,
                           resume_epoch=args.resume_epoch, resume_from=args.resume_from)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    if args.init_pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.init_pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    print(model)

    def loop(loader, step, epoch, prefix=''):  # Training or validation loop
        metrics = defaultdict(utils.HistoryMeter if prefix == 'train_' else utils.AverageMeter)
        tic()
        for i, batch in enumerate(loader):
            if args.cuda:
                batch = utils.cuda(batch)
            data_time = 1000 * toc(); tic()
            step_metrics = step(batch)
            step_metrics['data_time'] = data_time
            step_metrics['batch_time'] = 1000 * toc(); tic()
            for (k, v) in step_metrics.items():
                metrics[prefix + k].update(v, len(batch['input']))
            print(logging.str_metrics(metrics, iter=i, num_iters=len(loader), epoch=epoch, num_epochs=1))
        print(logging.str_metrics(metrics, epoch=epoch, num_epochs=1))
        toc()
        if prefix == 'val_':
            return OD((k, v.avg) for (k, v) in metrics.items())
        return OD((k, v.hist) for (k, v) in metrics.items())

    if args.validate_first and 0 not in metrics_history:
        model.eval()
        metrics_history[0] = loop(loaders['val'], validation_step, 0, 'val_')
        checkpoints.save_metrics(metrics_history)

    model.eval()  # freeze batch normalization
    metrics = loop(loaders['train'], training_step, 0, 'train_')

    if args.validate:
        model.eval()
        metrics.update(loop(loaders['val'], validation_step, 0, 'val_'))

        metrics_history[1] = metrics

    if not args.dry:
        utils.make_plots(metrics_history, args.expdir)
        checkpoints.save(model, 1, optimizers, metrics_history)


if __name__ == "__main__":
    parser = ArgumentParser(description="""GeM p exponent finetuning script for MultiGrain model, 
                                           computes the p exponent for a given input size by fine-tuning""",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500', help='experiment destination directory')
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    parser.add_argument('--resume-from', default='experiments/resnet50', help='resume checkpoint file/folder')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='base learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum in SGD')
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--images-per-class', default=50, type=int,
                        help='use a training subset of N images per class for the finetuning')
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    parser.add_argument('--pretrained-backbone', action='store_const', const='imagenet', help='use pretrained backbone')
    parser.add_argument('--no-validate-first', action='store_true', help='do not validate before training')
    parser.add_argument('--no-validate', action='store_true',
                        help='do not validate after training')
    parser.add_argument('--init-pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    parser.add_argument('--imagenet-path', default='data/ilsvrc2012', help='ImageNet data root')
    parser.add_argument('--preload-dir-imagenet', default=None,
                        help='preload imagenet in this directory (useful for slow networks')
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()

    run(args)
