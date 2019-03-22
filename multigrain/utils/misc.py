# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os


def ifmakedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

