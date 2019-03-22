# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.utils.data as data


class IdDataset(data.Dataset):
    """
    Return image id with getitem in dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        returns = self.dataset[index]
        return_dict = {}
        if not isinstance(returns, dict):
            return_dict['input'], return_dict['classifier_target'] = returns
        return_dict['instance_target'] = index
        return return_dict

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return "IdDataset(" + repr(self.dataset) + ")"