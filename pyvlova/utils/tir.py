# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable


def tir_store(producer, index, value):
    from tvm import tir
    if not isinstance(index, Iterable):
        index = [index]
    return tir.ProducerStore(producer, indices=list(index), value=value)


def tir_load(producer, index):
    from tvm import tir
    if not isinstance(index, Iterable):
        index = [index]
    return tir.ProducerLoad(producer=producer, indices=list(index))


def tir_imm(obj, dtype=None):
    from tvm import tir
    if isinstance(obj, tir.PrimExpr):
        return obj
    if isinstance(obj, bool):
        return tir.IntImm(dtype=dtype or 'bool', value=obj)
    if isinstance(obj, float):
        return tir.FloatImm(dtype=dtype or 'float32', value=obj)
    if isinstance(obj, int):
        return tir.IntImm(dtype=dtype or 'int32', value=obj)
    if isinstance(obj, str):
        return tir.StringImm(obj)
    assert False


def tir_cuda_shared_sync():
    import tvm
    from tvm import tir
    return tir.Call(None, 'tir.tvm_storage_sync', tvm.runtime.convert(['shared']))


def tir_thread_extent_attr(iter_var, extent=0, body=None):
    from tvm import tir
    extent = tir_imm(extent)
    return tir.AttrStmt(
        node=iter_var, attr_key='thread_extent',
        value=extent, body=body
    )
