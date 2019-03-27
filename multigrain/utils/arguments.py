# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse


def comma_separated(type, separator=','):
    def parser(inp):
        out = tuple(type(i) for i in inp.split(separator))
        if out == ('',):
            out = ()
        return out
    return parser


def float_in_range(begin, end):
    def parser(inp):
        inp = float(inp)
        if not begin <= inp <= end:
            raise argparse.ArgumentTypeError('Argument should be between {} and {}'.format(begin, end))
        return inp
    return parser


def compare_dicts(dict1, dict2, verbose=True):
    removed = []
    added = []
    changed = []
    for k in dict1:
        if k not in dict2:
            removed.append((k, dict1[k]))
        elif dict2[k] != dict1[k]:
            changed.append((k, dict1[k], dict2[k]))
    for k in dict2:
        if k not in dict2:
            added.append((k, dict2[k]))
    if verbose:
        if removed:
            print('removed keys:', ', '.join('{} ({})'.format(k, v) for (k, v) in removed))
        if added:
            print('added keys:', ', '.join('{} ({})'.format(k, v) for (k, v) in added))
        if changed:
            print('changed keys:', ', '.join('{} ({} -> {})'.format(k, v1, v2) for (k, v1, v2) in changed))
    return removed, added, changed
