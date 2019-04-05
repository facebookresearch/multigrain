# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Monkey-patches https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet_mobile.py
The source implementation cannot be applied to images of arbitrary shape
Here we add cropping operations before additions and concatenations to address this.
"""
from .pnasnet import shrink_sum, shrink_cat
import pretrainedmodels


def CellStem0_forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_comb_iter_2_right)

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = shrink_sum(x_comb_iter_3_right, x_comb_iter_1)

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_comb_iter_4_right)

        x_out = shrink_cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


def CellStem1_forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(shrink_cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_comb_iter_2_right)

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = shrink_sum(x_comb_iter_3_right, x_comb_iter_1)

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_comb_iter_4_right)

        x_out = shrink_cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


def FirstCell_forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(shrink_cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_left)

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = shrink_sum(x_comb_iter_3_left, x_comb_iter_3_right)

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_right)

        x_out = shrink_cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


def NormalCell_forward(self, x, x_prev):
    x_left = self.conv_prev_1x1(x_prev)
    x_right = self.conv_1x1(x)

    x_comb_iter_0_left = self.comb_iter_0_left(x_right)
    x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

    x_comb_iter_1_left = self.comb_iter_1_left(x_left)
    x_comb_iter_1_right = self.comb_iter_1_right(x_left)
    x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

    x_comb_iter_2_left = self.comb_iter_2_left(x_right)
    x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_left)

    x_comb_iter_3_left = self.comb_iter_3_left(x_left)
    x_comb_iter_3_right = self.comb_iter_3_right(x_left)
    x_comb_iter_3 = shrink_sum(x_comb_iter_3_left, x_comb_iter_3_right)

    x_comb_iter_4_left = self.comb_iter_4_left(x_right)
    x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_right)

    x_out = shrink_cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    return x_out


def ReductionCell0_forward(self, x, x_prev):
    x_left = self.conv_prev_1x1(x_prev)
    x_right = self.conv_1x1(x)

    x_comb_iter_0_left = self.comb_iter_0_left(x_right)
    x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

    x_comb_iter_1_left = self.comb_iter_1_left(x_right)
    x_comb_iter_1_right = self.comb_iter_1_right(x_left)
    x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

    x_comb_iter_2_left = self.comb_iter_2_left(x_right)
    x_comb_iter_2_right = self.comb_iter_2_right(x_left)
    x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_comb_iter_2_right)

    x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
    x_comb_iter_3 = shrink_sum(x_comb_iter_3_right, x_comb_iter_1)

    x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
    x_comb_iter_4_right = self.comb_iter_4_right(x_right)
    x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_comb_iter_4_right)

    x_out = shrink_cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    return x_out


def ReductionCell1_forward(self, x, x_prev):
    x_left = self.conv_prev_1x1(x_prev)
    x_right = self.conv_1x1(x)

    x_comb_iter_0_left = self.comb_iter_0_left(x_right)
    x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0 = shrink_sum(x_comb_iter_0_left, x_comb_iter_0_right)

    x_comb_iter_1_left = self.comb_iter_1_left(x_right)
    x_comb_iter_1_right = self.comb_iter_1_right(x_left)
    x_comb_iter_1 = shrink_sum(x_comb_iter_1_left, x_comb_iter_1_right)

    x_comb_iter_2_left = self.comb_iter_2_left(x_right)
    x_comb_iter_2_right = self.comb_iter_2_right(x_left)
    x_comb_iter_2 = shrink_sum(x_comb_iter_2_left, x_comb_iter_2_right)

    x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
    x_comb_iter_3 = shrink_sum(x_comb_iter_3_right, x_comb_iter_1)

    x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
    x_comb_iter_4_right = self.comb_iter_4_right(x_right)
    x_comb_iter_4 = shrink_sum(x_comb_iter_4_left, x_comb_iter_4_right)

    x_out = shrink_cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
    return x_out


def nasnetamobile(*args, **kwargs):
    pretrainedmodels.models.nasnet_mobile.CellStem0.forward = CellStem0_forward
    pretrainedmodels.models.nasnet_mobile.CellStem1.forward = CellStem1_forward
    pretrainedmodels.models.nasnet_mobile.FirstCell.forward = FirstCell_forward
    pretrainedmodels.models.nasnet_mobile.NormalCell.forward = NormalCell_forward
    pretrainedmodels.models.nasnet_mobile.ReductionCell0.forward = ReductionCell0_forward
    pretrainedmodels.models.nasnet_mobile.ReductionCell1.forward = ReductionCell1_forward
    model = pretrainedmodels.models.nasnetamobile(*args, **kwargs)
    return model
