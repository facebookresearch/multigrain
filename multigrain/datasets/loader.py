# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from shutil import copyfile
import os.path as osp
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from multigrain.utils import ifmakedirs


def loader(path):
    return Image.open(path).convert('RGB')


def preloader(dataset_root, preload_dir):
    def this_loader(path):
        dest_path = osp.join(preload_dir, osp.relpath(path, dataset_root))
        ifmakedirs(osp.dirname(dest_path))

        if not osp.isfile(dest_path):
            copyfile(path, dest_path)

        image = loader(dest_path)
        return image
    return this_loader