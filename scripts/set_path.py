# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Convenience script adding MultiGrain project to python path without mandating installation
Simply adds parent folder to this script to path
"""
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(__file__))))
