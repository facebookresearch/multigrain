# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils import data
import os.path as osp
from .loader import loader as default_loader


class ListDataset(data.Dataset):
    """
    Unlabelled images dataset from list of images and root
    """

    def __init__(self, root, imagelist, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.imgs = imagelist
        if isinstance(imagelist, str):
            self.imgs = []
            if not osp.isfile(imagelist):
                raise FileNotFoundError('Image list not found at {}'.format(imagelist))
            with open(imagelist) as f:
                for im in f:
                    im = im.strip()
                    if not im: continue
                    self.imgs.append(im)
        self.loader = loader

    def __getitem__(self, idx):
        image = self.loader(osp.join(self.root, self.imgs[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.imgs)
