# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import re

import isl


def get_named_tuples(obj):
    obj = str(obj)
    res = re.findall(r'(\w+)\[(.*?)\]', str(obj))
    for i in range(len(res)):
        res[i] = list(res[i])
        res[i][1] = list(map(str.strip, res[i][1].split(',')))
    return res


def get_unnamed_tuples(obj):
    obj = str(obj)
    res = re.findall(r'\[(.*?)\]', str(obj))
    for i in range(len(res)):
        res[i] = list(map(str.strip, res[i].split(',')))
    return res


def structure_named_fixed_box(box: isl.fixed_box):
    (name, size), *_ = get_named_tuples(box.size())
    size = list(map(int, size))
    offset = box.offset()
    return name, size, offset


def structure_unnamed_fixed_box(box: isl.fixed_box):
    size, *_ = get_unnamed_tuples(box.size())
    size = list(map(int, size))
    offset = box.offset()
    return size, offset


def map_out_constant_dim(obj):
    if hasattr(obj, 'coalesce'):
        obj = obj.coalesce()
    old, *_ = get_unnamed_tuples(obj)
    new = filter(lambda x: not x.isdigit(), old)
    return isl.map(f'{{ [{", ".join(old)}] -> [{", ".join(new)}] }}')
