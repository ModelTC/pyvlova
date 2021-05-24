# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import re

from .._ext import isl


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


isl.isl.isl_map_transitive_closure.restype = isl.c_void_p
isl.isl.isl_map_transitive_closure.argtypes = [isl.c_void_p, isl.c_void_p]
isl.isl.isl_union_map_transitive_closure.restype = isl.c_void_p
isl.isl.isl_union_map_transitive_closure.argtypes = [isl.c_void_p, isl.c_void_p]


def vanilla_isl_transitive_closure(umap):
    umap = umap.coalesce()
    if isinstance(umap, isl.map):
        ptr = isl.isl.isl_map_copy(umap.ptr)
        ptr = isl.isl.isl_map_transitive_closure(ptr, None)
    else:
        ptr = isl.isl.isl_union_map_copy(umap.ptr)
        ptr = isl.isl.isl_union_map_transitive_closure(ptr, None)
    return type(umap)(ctx=umap.ctx, ptr=ptr).coalesce()


def binary_exp_transitive_closure(umap):
    umap = umap.coalesce()
    identity = umap.domain().identity()
    if isinstance(umap, isl.map):
        umap = umap.add_map(identity)
    else:
        umap = umap.union(identity)
    umap = umap.coalesce()

    for _ in range(64):
        if umap.n_map() > 16 or len(str(umap)) > 4096:
            return None
        doubled = umap.apply_range(umap).coalesce()
        if doubled.is_equal(umap):
            return umap
        umap = doubled

    return None


def transitive_closure(umap):
    res = binary_exp_transitive_closure(umap)
    if res is None:
        return vanilla_isl_transitive_closure(umap)
    return res
