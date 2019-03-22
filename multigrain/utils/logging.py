# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import yaml
from collections import OrderedDict as OD
import os
import os.path as osp


def num_fmt(num, n=1):
    """format digits with n-significant digits"""
    if isinstance(num, int):
        return str(num)
    # round to n significant digits using scientific notation
    num = float(('{:.' + str(n - 1) + 'e}').format(num))
    return str(int(num) if num.is_integer() else num)

# https://stackoverflow.com/a/21912744/805502
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OD):
    if not hasattr(stream, 'read'):  # filename instead of stream
        stream = open(stream, 'r')
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    loaded = yaml.load(stream, OrderedLoader)
    stream.close()
    return loaded


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OD, _dict_representer)
    if hasattr(stream, 'write'):
        dumped = yaml.dump(data, stream, OrderedDumper, **kwds)
    else:  # filename instead of file
        dumped = yaml.dump(data, None, OrderedDumper, **kwds)
        print_file(dumped, stream)
    return dumped


def str_metrics(metrics, epoch, num_epochs, iter=None, num_iters=None):
    str = '[Ep {}/{}] '.format(epoch, num_epochs)
    if iter is not None:
        str += '({}/{}) '.format(iter, num_iters)
    metricstr = []
    count = len(str)
    for name, value in metrics.items():
        if iter is not None:
            new = "{} {} ({})".format(name, num_fmt(value.val, 3), num_fmt(value.avg, 3))
        else:
            new = "{}_avg {}".format(name, num_fmt(value.avg, 3))
        if count + len(new) > 90:
            count = len(new)
            new = '\n' + new
        else:
            count += len(new)
        metricstr.append(new)
    str += ', '.join(metricstr)
    return str


def print_file(str, filename, safe_overwrite=True):
    """
    Write a string to a file;
    if the file exists and safe_overwrite is true, do a safe overwriting.
    """
    tmp = None
    if osp.isfile(filename) and safe_overwrite:
        tmp = osp.join(osp.dirname(filename), osp.basename(filename) + '.old')
        os.rename(filename, tmp)
    with open(filename, 'w') as f:
        f.write(str)
    if tmp is not None:
        os.remove(tmp)
