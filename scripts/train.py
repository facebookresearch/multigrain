# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms, transforms_list
from multigrain.lib import get_multigrain, RASampler
from multigrain.datasets import IN1K, IdDataset, default_loader, preloader
from multigrain import utils
from multigrain.modules import MultiCriterion, MultiOptim, MarginLoss, SampledMarginLoss
from multigrain.backbones import backbone_list

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os.path as osp
from collections import defaultdict
from collections import OrderedDict as OD
tic, toc = utils.Tictoc()


def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)
    argfile = osp.join(args.expdir, 'train_args.yaml')

    if osp.isfile(argfile):
        oldargs = yaml.load(open(argfile))
        if oldargs is not None and oldargs != args.__dict__:
            print('WARNING: Changed configuration keys compared to stored experiment')
            utils.arguments.compare_dicts(oldargs, args.__dict__, verbose=True)

    args.cuda = not args.no_cuda
    args.validate_first = not args.no_validate_first

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    transforms = get_transforms(IN1K, args.input_size, args.augmentation, args.backbone)
    datas = {}
    for split in ('train', 'val'):
        imload = preloader(args.imagenet_path, args.preload_dir_imagenet) if args.preload_dir_imagenet else default_loader
        datas[split] = IdDataset(IN1K(args.imagenet_path, split, transform=transforms[split], loader=imload))
    loaders = {}
    loaders['train'] = DataLoader(datas['train'],
                       batch_sampler=RASampler(len(datas['train']), args.batch_size, args.repeated_augmentations,
                                               args.epoch_len_factor, shuffle=True, drop_last=False),
                       num_workers=args.workers, pin_memory=True)
    loaders['val'] = DataLoader(datas['val'], batch_size=args.batch_size, shuffle=args.shuffle_val,
                                num_workers=args.workers, pin_memory=True)

    model = get_multigrain(args.backbone, p=args.pooling_exponent, include_sampling=not args.global_sampling,
                           pretrained_backbone=args.pretrained_backbone)
    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, args.pooling_exponent))
    print(model)

    cross_entropy = torch.nn.CrossEntropyLoss()
    cross_entropy_criterion = (cross_entropy,
                               ('classifier_output', 'classifier_target'),
                               args.classif_weight)
    if args.global_sampling:
        margin = SampledMarginLoss(margin_args=dict(beta_init=args.beta_init))
        beta = margin.margin.beta
        margin_criterion = (margin,
                            ('normalized_embedding', 'instance_target'),
                            1.0 - args.classif_weight)
    else:
        margin = MarginLoss(args.beta_init)
        beta = margin.beta
        margin_criterion = (margin,
                            ('anchor_embeddings', 'negative_embeddings', 'positive_embeddings'),
                            1.0 - args.classif_weight)

    extra = {'beta': beta}

    criterion = MultiCriterion(dict(cross_entropy=cross_entropy_criterion, margin=margin_criterion),
                               skip_zeros=(args.repeated_augmentations == 1))

    if args.cuda:
        criterion = utils.cuda(criterion)
        model = utils.cuda(model)

    optimizers = OD()
    optimizers['backbone'] = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizers['margin_beta'] = SGD([beta], lr=args.learning_rate * args.beta_lr, momentum=args.momentum)
    optimizers = MultiOptim(optimizers)
    optimizers.set_base_lr()

    if args.cuda:
        model = nn.DataParallel(model)

    def set_learning_rate(epoch):
        factor = 1.0
        for (drop, gamma) in zip(args.lr_drops_epochs, args.lr_drops_factors):
            if epoch > drop:
                factor *= gamma
        optimizers.lr_multiply(factor)

    batches_accumulated = 0

    def training_step(batch):
        nonlocal batches_accumulated
        if batches_accumulated == 0:
            optimizers.zero_grad()

        output_dict = model(batch['input'], batch['instance_target'])
        output_dict['classifier_target'] = batch['classifier_target']
        loss_dict = criterion(output_dict)
        top1, top5 = utils.accuracy(output_dict['classifier_output'].data, output_dict['classifier_target'].data, topk=(1, 5))

        loss_dict['loss'].backward()
        batches_accumulated += 1

        if batches_accumulated == args.gradient_accum:
            mag = {}
            for (name, p) in model.named_parameters():
                mag[name] = p.grad.norm().item()
            optimizers.step()
            batches_accumulated = 0

        return_dict = OD()
        for key in ['cross_entropy', 'margin', 'loss']:
            if key in loss_dict:
                return_dict[key] = loss_dict[key].item()
        return_dict['beta'] = beta.item()
        return_dict['top1'] = top1
        return_dict['top5'] = top5

        return return_dict

    def validation_step(batch):
        with torch.no_grad():
            output_dict = model(batch['input'])
            target = batch['classifier_target']
            xloss = cross_entropy(output_dict['classifier_output'], target)
            top1, top5 = utils.accuracy(output_dict['classifier_output'], target, topk=(1, 5))

        return OD([
            ('cross_entropy', xloss.item()),
            ('top1', top1),
            ('top5', top5),
        ])

    metrics_history = OD()

    checkpoints = utils.CheckpointHandler(args.expdir, args.save_every)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        begin_epoch, loaded_extra = checkpoints.resume(model, optimizers, metrics_history, args.resume_epoch, args.resume_from)
        if 'beta' in loaded_extra:
            beta.data.copy_(loaded_extra['beta'])
        else:
            print('(re)initialized beta to {}'.format(beta.item()))
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    def loop(loader, step, epoch, prefix=''):  # Training or validation loop
        metrics = defaultdict(utils.AverageMeter)
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
            print(logging.str_metrics(metrics, iter=i, num_iters=len(loader), epoch=epoch, num_epochs=args.epochs))
        print(logging.str_metrics(metrics, epoch=epoch, num_epochs=args.epochs))
        toc()
        return OD((k, v.avg) for (k, v) in metrics.items())

    if args.validate_first and begin_epoch == 0 and 0 not in metrics_history:
        model.eval()
        metrics_history[0] = loop(loaders['val'], validation_step, begin_epoch, 'val_')
        checkpoints.save_metrics(metrics_history)

    for epoch in range(begin_epoch, args.epochs):
        set_learning_rate(epoch)

        batches_accumulated = 0
        model.train()
        metrics = loop(loaders['train'], training_step, epoch, 'train_')

        model.eval()
        metrics.update(loop(loaders['val'], validation_step, epoch, 'val_'))

        metrics_history[epoch + 1] = metrics

        if not args.dry:
            utils.make_plots(metrics_history, args.expdir)
            checkpoints.save(model, epoch + 1, optimizers, metrics_history, extra)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for MultiGrain models", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--expdir', default='experiments/resnet50', help='experiment directory')
    parser.add_argument('--shuffle-val', action='store_true', help='shuffle val. dataset')
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    parser.add_argument('--resume-from', default=None, help='resume checkpoint file/folder (default same as experiment)')
    parser.add_argument('--learning-rate', default=0.2, type=float, help='base learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum in SGD')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch-size', default=512, type=int, help='batch size')
    parser.add_argument('--gradient-accum', default=1, type=int, help='number of batches in one backward pass')
    parser.add_argument('--repeated-augmentations', default=1, type=int, help='repetitions in repeated augmentations')
    parser.add_argument('--augmentation', default='full', choices=transforms_list, help='augmentation type')
    parser.add_argument('--epochs', default=120, type=int, help='training epochs')
    parser.add_argument('--epoch-len-factor', default=2.0, type=float,
                        help='multiplier for epoch size')
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list,
                        help='backbone architecture')
    parser.add_argument('--pretrained-backbone', action='store_const', const='imagenet', help='use pretrained backbone')
    parser.add_argument('--lr-drops-epochs', default=[30, 60, 90],
                        type=utils.arguments.comma_separated(int), help='learning rate drops epochs')
    parser.add_argument('--lr-drops-factors', default=[0.1, 0.1, 0.1],
                        type=utils.arguments.comma_separated(float), help='learning rate drops multipliers')
    parser.add_argument('--no-validate-first', action='store_true', help='do not validate before training')
    parser.add_argument('--pooling-exponent', default=1.0, type=float, help='exponent in GeM pooling')
    parser.add_argument('--beta-init', default=1.2, type=float, help='initial value for beta in margin loss')
    parser.add_argument('--beta-lr', default=1.0, type=float,
                         help='learning rate for beta (relative to base learning rate)')
    parser.add_argument('--global-sampling', action='store_true',
                         help='use a global weighted sampling instead of per-gpu (useful for small batches)')
    parser.add_argument('--classif-weight', default=1.0, type=utils.arguments.float_in_range(0, 1),
                         help='weighting parameter for the loss, between 0 (only margin) and 1 (only cross-entropy)')
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    parser.add_argument('--save-every', default=10, type=int, help='epoch backup interval (last epoch will always be saved)')
    parser.add_argument('--imagenet-path', default='data/ilsvrc2012', help='ImageNet data root')
    parser.add_argument('--preload-dir-imagenet', default=None,
                        help='preload imagenet in this directory (useful for slow networks')
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()
    if args.repeated_augmentations == 1:
        if args.classif_weight != 1.0:
            raise ValueError('Margin loss in undefined for repeated_augmentations == 1; set --classif-weight=1.0')
        # No sampling is actually computed in this case, but the implementation requires the following:
        args.global_sampling = True


    run(args)
