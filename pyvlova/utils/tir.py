from typing import Iterable

from tvm import te, tir


def tir_store(tensor: te.Tensor, index: Iterable[tir.PrimExpr], value: tir.PrimExpr) -> tir.Provide:
    return tir.Provide(
        func=tensor.op, value_index=tensor.value_index, args=list(index),
        value=value
    )


def tir_load(tensor: te.Tensor, index: Iterable[tir.PrimExpr]) -> tir.Call:
    return tir.Call(
        func=tensor.op, value_index=tensor.value_index, args=list(index),
        call_type=3, name=tensor.name, dtype=tensor.dtype
    )


def tir_imm(obj, dtype=None) -> tir.PrimExpr:
    if isinstance(obj, bool):
        return tir.IntImm(dtype=dtype or 'bool', value=obj)
    if isinstance(obj, float):
        return tir.FloatImm(dtype=dtype or 'float32', value=obj)
    if isinstance(obj, int):
        return tir.IntImm(dtype=dtype or 'int32', value=obj)
    if isinstance(obj, str):
        return tir.StringImm(obj)
    assert False