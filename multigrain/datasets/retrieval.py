# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.utils.data as data
from glob import glob
import os
import os.path as osp
from .loader import loader
from collections import OrderedDict as OD
from torchvision.datasets.utils import download_url
from multigrain.utils.misc import ifmakedirs
import zipfile
import tarfile
import yaml


class DownloadableDataset(data.Dataset):
    URLS = []
    NUM_FILES = []

    def __init__(self, root):
        self.root = root

    def _check_exists(self):
        for pattern, number in self.NUM_FILES:
            if len(glob(osp.join(self.root, pattern))) != number:
                return False
        return True

    def download(self, remove_finished=True):
        if self._check_exists():
            return

        ifmakedirs(self.root)

        for url in self.URLS:
            subdest = ''
            if isinstance(url, tuple):
                subdest, url = url
            # download file
            filename = url.rpartition('/')[2]
            file_path = osp.join(self.root, filename)
            download_url(url, root=self.root, filename=filename, md5=None)
            dest = osp.join(self.root, subdest)
            ifmakedirs(dest)
            if filename.endswith('.zip'):
                self.extract_zip(zip_path=file_path, dest=dest, remove_finished=remove_finished)
            elif filename.endswith('.tar') or filename.endswith('.tar.gz'):
                self.extract_tar(file_path, dest=dest, remove_finished=remove_finished)
            else:
                raise ValueError('File {}: has unknown extension'.format(filename))
        print('Done!')

    @staticmethod
    def extract_zip(zip_path, dest, remove_finished=True):
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()
        if remove_finished:
            os.unlink(zip_ref)

    @staticmethod
    def extract_tar(fname, dest, remove_finished=True):
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(path=dest)
            tar.close()
        elif (fname.endswith("tar")):
            tar = tarfile.open(fname, "r:")
            tar.extractall(path=dest)
            tar.close()
        if remove_finished:
            os.unlink(fname)


class UKBench(DownloadableDataset):
    """UKBench dataset."""

    URLS = ['https://archive.org/download/ukbench/ukbench.zip']
    NUM_FILES = [('*.jpg', 10200)]

    def __init__(self, root, transform=None, download=False):
        self.root = root
        if download:
            self.download()
        images = glob(osp.join(self.root, '*.jpg'))
        grouped = []
        for i in range(0, len(images), 4):
            grouped.append([images[i], images[i + 1], images[i + 2], images[i + 3]])
        self.imgs = []
        self.class_groups = {}
        for c, G in enumerate(grouped):
            for im in G:
                self.imgs.append((im, c))
                self.class_groups.setdefault(c, []).append(len(self.imgs) - 1)
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class Holidays(DownloadableDataset):
    """Holidays dataset."""
    URLS = ['ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz',
            'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz']
    NUM_FILES = [(osp.join('jpg', '*.jpg'), 1491)]

    def __init__(self, root, transform=None, rotated=True, download=False):
        self.root = root
        if download:
            self.download()
        images = sorted(glob(osp.join(self.root, 'jpg', '*.jpg')))
        cur_group = [images[0]]
        grouped = []
        for i in images[1:]:
            if int(osp.basename(i[:-len('.jpg')])) % 100:
                cur_group.append(i)
            else:
                grouped.append(cur_group)
                cur_group = [i]
        self.imgs = []
        self.class_groups = {}
        for c, G in enumerate(grouped):
            for im in G:
                self.imgs.append((im, c))
                self.class_groups.setdefault(c, []).append(len(self.imgs) - 1)
        self.loader = loader
        self.transform = transform
        self.rotated = None
        if rotated:
            self.rotated = yaml.load(open(osp.join(osp.dirname(__file__), 'holidays-rotate.yaml')))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.rotated is not None:
            rotation = self.rotated.get(osp.basename(path), 0)
            if rotation != 0:
                sample = sample.rotate(-rotation, expand=True)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class CopyDays(DownloadableDataset):
    """CopyDays dataset."""
    URLS = ['http://pascal.inrialpes.fr/data/holidays/copydays_original.tar.gz',
            'http://pascal.inrialpes.fr/data/holidays/copydays_crop.tar.gz',
            'http://pascal.inrialpes.fr/data/holidays/copydays_jpeg.tar.gz',
            ('strong', 'http://pascal.inrialpes.fr/data/holidays/copydays_strong.tar.gz')]
    NUM_FILES = [('*.jpg', 157), ('*/*.jpg', 229), ('*/*/*.jpg', 2826)]

    def __init__(self, root, subset=None, transform=None):
        self.root = root
        self.download()

        self.distractors = distractors
        self.num_distractors = num_distractors
        self.subset = subset

        avail = OD()
        for x in os.walk(root):
            for filename in x[2]:
                id = int(filename.split('.')[0])
                transf = osp.relpath(x[0], root)
                id = (id // 100) * 100
                if transf == '.':
                    transf = ''
                avail.setdefault(id, OD())[transf] = osp.join(x[0], filename)

        if not avail:
            raise ValueError("Dataset not found in {}".format(root))

        transfs = []
        self.images = []
        self.class_groups = []
        for i, id in enumerate(avail):
            im = {'variant': -1,
                  'input': avail[id][''],
                  'target': i}
            self.images.append(im)
            cur_group = [len(self.images) - 1]
            for transf in avail[id]:
                if not transf:
                    continue
                if self.subset is not None and transf not in self.subset:
                    continue
                if transf not in transfs:
                    transfs.append(transf)
                im = {'variant': transfs.index(transf),
                      'input': avail[id][transf],
                      'target': i}
                self.images.append(im)
                cur_group.append(len(self.images) - 1)
            self.class_groups.append(cur_group)

        self.gen_distractors()
        # if num_distractors is not None and len(self.distractors) > num_distractors:
        #     self.distractors = self.distractors[:num_distractors]
        self.transfs = transfs
        self.loader = loader
        self.transform = transform

    def gen_distractors(self):
        self.distractor_list = []  # glob(distractors)
        for dirpath, subdirs, files in os.walk(self.distractors):
            for x in files:
                if x.endswith('.jpg'):
                    self.distractor_list.append(osp.join(dirpath, x))
                if len(self.distractor_list) >= self.num_distractors:
                    return

    def __len__(self):
        return len(self.images) + len(self.distractor_list)

    def __getitem__(self, index):
        if index < len(self.images):
            return_dict = self.images[index].copy()
        else:
            index -= len(self.images)
            return_dict = {'transf': -1,
                           'input': self.distractor_list[index],
                           'target': -1}
        im = self.loader(return_dict['input'])
        if self.transform is not None:
            im = self.transform(im)
        return_dict['input'] = im
        return return_dict

#
# class Distractors(data.Dataset):
#     """
#     Distractor dataset
#     Finds images in subfolders recursively
#     """
#
#
#
#
# class Distracted(data.Dataset):
#     """
#     Distracts a retrieval dataset with distractors
#     """
#     def __init__(self, dataset, distractors):
