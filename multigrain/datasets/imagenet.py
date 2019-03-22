# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Imagenet is either distributed along with a devkit to get the validation labels,
 or with the validation set reorganized into different subsets.
Here we support both.
Keeps an index of the images for fast initialization.
"""

import torch
from torch.utils import data
import os
from os import path as osp
import numpy as np
from .loader import loader as default_loader
from multigrain.utils import ifmakedirs


class IN1K(data.Dataset):
    """
    ImageNet 1K dataset
    Classes numbered from 0 to 999 inclusive
    Can deal both with ImageNet original structure and the common "reorganized" validation dataset
    Caches list of files for faster reloading.
    """
    NUM_CLASSES = 1000
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EIG_VALS = [0.2175, 0.0188, 0.0045]
    EIG_VECS = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])

    def __init__(self, root, split='train', transform=None, force_reindex=False, loader=default_loader):
        self.root = root
        self.transform = transform
        self.split = split
        cachefile = 'data/IN1K-' + split + '-cached-list.pth'
        self.classes, self.class_to_idx, self.imgs, self.labels, self.images_subdir = self.get_dataset(cachefile, force_reindex)
        self.loader = loader

    def get_dataset(self, cachefile=None, force_reindex=False):
        if osp.isfile(cachefile) and not force_reindex:
            print('Loaded IN1K {} dataset from cache: {}...'.format(self.split, cachefile))
            return torch.load(cachefile)

        print('Indexing IN1K {} dataset...'.format(self.split), end=' ')
        for images_subdir in [self.split, 'ILSVRC2012_img_' + self.split]:
            if osp.isdir(osp.join(self.root, images_subdir)):
                break
        else:
            raise ValueError('Split {} not found'.format(self.split))
        self.images_subdir = images_subdir
        subfiles = os.listdir(osp.join(self.root, images_subdir))
        if osp.isdir(osp.join(self.root, images_subdir, subfiles[0])):  # ImageFolder
            classes = [folder for folder in subfiles if folder.startswith('n')]
            classes.sort()
            class_to_idx = {c: i for (i, c) in enumerate(classes)}
            imgs = []
            labels = []
            for label in classes:
                label_images = os.listdir(osp.join(self.root, images_subdir, label))
                label_images.sort()
                imgs.extend([osp.join(label, i) for i in label_images])
                labels.extend([class_to_idx[label] for _ in label_images])
        else:  # DevKit
            try:
                import mat4py
            except ImportError:
                print('Install mat4py to discover classes from meta.mat')
                raise
            synsets = mat4py.loadmat(osp.join(self.root,
                                                   'ILSVRC2012_devkit_t12',
                                                   'data',
                                                   'meta.mat'))['synsets']

            ilsvrc_label_to_wnid = {label: wn
                                     for (wn, label) in zip(synsets['WNID'],
                                                            synsets['ILSVRC2012_ID'])
                                     if label <= self.NUM_CLASSES}
            classes = sorted(ilsvrc_label_to_wnid.values())
            class_to_idx = {c: i for (i, c) in enumerate(classes) if i < self.NUM_CLASSES}
            imgs = sorted(subfiles)
            ilsvrc_labels = np.loadtxt(osp.join(self.root,
                                                'ILSVRC2012_devkit_t12',
                                                'data',
                                                'ILSVRC2012_validation_ground_truth.txt'
                                                ), dtype=int)
            labels = [class_to_idx[ilsvrc_label_to_wnid[l]] for l in ilsvrc_labels]

            sort_by_label = sorted(zip(labels, imgs))
            labels, imgs = list(zip(*sort_by_label))
        print('OK!', end='')
        returns = (classes, class_to_idx, imgs, labels, images_subdir)
        if cachefile is not None:
            ifmakedirs(osp.dirname(cachefile))
            torch.save(returns, cachefile)
            print(' cached to', cachefile)
        print()
        return returns
        
    def __getitem__(self, idx):
        image = self.loader(osp.join(self.root, self.images_subdir, self.imgs[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.labels[idx])
    
    def __len__(self):
        return len(self.imgs)
    
    def __repr__(self):
        return "IN1K(root='{}', split='{}')".format(self.root, self.split)
