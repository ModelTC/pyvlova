from __future__ import annotations

from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import reduce
from types import FunctionType
from typing import Dict, List, Optional, Tuple, Set, Iterable

import sympy
import tvm
import isl
from tvm import te, tir

from pyvlova.utils import Mode, tir_load, tir_store, tir_imm, sizeof
from pyvlova.codegen.sympy2isl import parse_sympy_to_isl_repr


trace_mode = Mode()


class _EffectiveOpRecorder(object):
    def __init__(self):
        self.record = list()

    def push(self):
        self.record.append(list())
        return self.record[-1]

    def pop(self):
        return self.record.pop(-1)

    @contextmanager
    def recording(self):
        r = self.push()
        yield r
        self.pop()

    def __call__(self, x):
        self.record[-1].append(x)

    def to_stmt(self, *args, **kwargs):
        assert trace_mode.mode
        return getattr(self, 'to_stmt_' + trace_mode.mode)(*args, **kwargs)

    def to_stmt_tvm(self):
        assert self.record
        record = self.record[-1]
        if len(record) <= 1:
            res = record[0]
        else:
            res = tir.SeqStmt(record)
        return res

    def get_tensor_access(self, stmt_isl_repr) -> Tuple[Dict[str, Dict[str, isl.union_map]], Dict[str, Set[Tensor]]]:
        isl_mapping = defaultdict(lambda: defaultdict(lambda: isl.union_map('{}')))
        vanilla = defaultdict(set)
        record = self.record[-1]
        for t, tensor, ind in record:
            ind = ', '.join(map(str, ind))
            new_map = isl.union_map(f'{{ {stmt_isl_repr} -> {tensor.name}[{ind}] }}')
            isl_mapping[t][tensor.name] = isl_mapping[t][tensor.name].union(new_map)
            vanilla[t].add(tensor)
        return isl_mapping, vanilla


record_effective_op = _EffectiveOpRecorder()


class Tensor(object):
    def __init__(self, name, shape, offset=None, dtype='float32'):
        if offset is None:
            offset = [tir_imm(0)] * len(shape)
        self.name = name
        self.shape = list(shape)
        self.offset = list(offset)
        self.dtype = dtype
        self.te_tensor = te.placeholder(self.shape, self.dtype, self.name)

    def to_isl_set(self):
        keys = ['i%d' % i for i in range(len(self.shape))]
        constraints = [
            f'{self.offset[i]} <= {keys[i]} < {self.offset[i] + self.shape[i]}'
            for i in range(len(self.shape))
        ]
        s = isl.set(f'{{ {self.name}[{", ".join(keys)}] : {" and ".join(constraints)} }}')
        return s

    def build_tir_realize(self, scope='', body=None):
        def realize(b):
            tensor = self.te_tensor
            # noinspection PyTypeChecker
            return tir.AttrStmt(
                node=tensor.op, attr_key='realize_scope', value=tir_imm(scope),
                body=tir.Realize(
                    func=tensor.op, value_index=tensor.value_index, dtype=tensor.dtype,
                    bounds=list(map(tvm.ir.Range.make_by_min_extent, self.offset, tensor.shape)),
                    condition=tir_imm(True),
                    body=tir.ProducerConsumer(
                        func=tensor.op, is_producer=True,
                        body=b
                    )
                )
            )
        if body is None:
            return realize
        return realize(body)

    @property
    def size_in_bytes(self):
        return reduce(int.__mul__, self.shape) * sizeof(self.dtype)

    def getitem_tvm(self, key):
        assert len(key) == len(self.shape)
        return tir_load(self.te_tensor, key)

    def getitem_tensor_access(self, key):
        assert len(key) == len(self.shape)
        key = list(map(parse_sympy_to_isl_repr, key))
        record_effective_op(('read', self, key))
        return sympy.var(self.name + '_item')

    def __getitem__(self, key):
        assert trace_mode.mode
        if not isinstance(key, Iterable):
            key = [key]
        else:
            key = list(key)
        return getattr(self, 'getitem_' + trace_mode.mode)(key)

    def setitem_tvm(self, key, value):
        assert len(key) == len(self.shape)
        if isinstance(value, (int, float, str, bool)):
            value = tir_imm(value)
        record_effective_op(tir_store(self.te_tensor, key, value))

    # noinspection PyUnusedLocal
    def setitem_tensor_access(self, key, value):
        assert len(key) == len(self.shape)
        key = list(map(parse_sympy_to_isl_repr, key))
        record_effective_op(('write', self, key))

    def __setitem__(self, key, value):
        if not isinstance(key, Iterable):
            key = [key]
        else:
            key = list(key)
        assert trace_mode.mode
        return getattr(self, 'setitem_' + trace_mode.mode)(key, value)


TensorTableItem = namedtuple('TensorTableItem', ['scope', 'tensor'])


class TensorTable(object):
    def __init__(self):
        self.table: Dict[str, Tensor] = dict()
        self.scoped_stack: Dict[str, List[TensorTableItem]] = dict()

    def add_tensor(self, *args, factory=Tensor, **kwargs):
        t = factory(*args, **kwargs)
        self[t.name] = t
        return t

    def del_tensor(self, name):
        del self[name]

    def __iter__(self):
        return iter(self.table.values())

    def __getitem__(self, key):
        if key in self.scoped_stack:
            return self.scoped_stack[key][-1].tensor
        return self.table[key]

    def __setitem__(self, key, value):
        self.table[key] = value
        self.scoped_stack[key] = [TensorTableItem('', value)]

    def __delitem__(self, key):
        del self.table[key]
        del self.scoped_stack[key]

    def __contains__(self, key):
        return key in self.table

    @contextmanager
    def scoped(self, name: str, scope: str, **kwargs):
        tensor = self.push_scoped(name, scope, **kwargs)
        yield scope, tensor
        self.pop_scoped(name)

    def push_scoped(self, name: str, scope: str, tensor: Optional[Tensor] = None, factory=Tensor, shape=None, **kwargs):
        if tensor is None:
            scoped_name = f'{name}.{scope}'
            tensor = factory(scoped_name, shape, **kwargs)
        self.scoped_stack[name].append(TensorTableItem(scope, tensor))
        return self.scoped_stack[name][-1]

    def pop_scoped(self, name: str):
        scope, tensor = self.scoped_stack[name].pop(-1)
        if not self.scoped_stack[name]:
            del self.scoped_stack[name]
        return tensor


class IterVarTable(object):
    def __init__(self):
        self.vars = dict()
        self.var_stack = []

    def __getitem__(self, key):
        if key in self.vars:
            return self.vars[key]
        return self.vars[key]

    def __contains__(self, key):
        return key in self.vars

    @property
    def top(self):
        return self.var_stack[-1]

    def push(self, name=None, var=None):
        if name is None:
            name = f'_i{len(self.var_stack)}'
        if var is None:
            var = tir.Var(name=name, dtype='int32')
        self.var_stack.append((var, name))
        self.vars[name] = var
        return var

    def pop(self):
        var, name = self.var_stack.pop()
        del self.vars[name]
        return var

    @contextmanager
    def var(self, name=None, var=None):
        v = self.push(name, var)
        yield v
        self.pop()


class CUDAIterVarTable(IterVarTable):
    def __init__(self):
        super().__init__()
        self.axis_cnt = defaultdict(int)
        self.var_extent = defaultdict(lambda: defaultdict(lambda: 1))
        self.axis_extent = defaultdict(lambda: 1)
        self.axis_idx = defaultdict(lambda: tir_imm(0))

    @contextmanager
    def axis(self, name, extent):
        with self.var(name) as v:
            self.axis_extent[name] *= extent
            self.axis_idx[name] = self.axis_idx[name] * extent + v
            self.var_extent[name][str(v.var)] = extent
            yield v
            del self.var_extent[name][str(v.var)]
            self.axis_idx[name] = (self.axis_idx[name] - v) // extent
            self.axis_extent[name] //= extent

    def push(self, name=None, var=None):
        k = self.axis_cnt[name]
        self.axis_cnt[name] += 1
        name = f'{name}.{chr(ord("x") + k)}'
        if var is None:
            var = te.thread_axis(name)
        return super().push(name, var)

    def pop(self):
        var = super().pop()
        axis, _ = var.var.name.split('.', 1)
        self.axis_cnt[axis] -= 1
        return var


class Statement(object):
    @classmethod
    def from_calc(cls, func: FunctionType) -> Statement:
        return cls(func.__name__, func.__code__.co_argcount - 1, func)

    def __init__(self, name, dim, calc, tensor_table=None):
        self.name = name
        self.dim = dim
        self.calc = calc
        self.access = None
        self.tensor_table = tensor_table

    def to_stmt(self, tensor_table, *args):
        assert len(args) == self.dim
        self.calc(tensor_table, *args)
        return record_effective_op.to_stmt()

    def to_tvm(self, tensor_table, *args):
        with trace_mode.under('tvm'), record_effective_op.recording():
            return self.to_stmt(tensor_table, *args)

    def get_access(self, tensor_table=None):
        if tensor_table is None:
            tensor_table = self.tensor_table
        if self.access is None:
            assert tensor_table is not None
            with trace_mode.under('tensor_access'), record_effective_op.recording():
                args = [sympy.var('i%d' % i) for i in range(self.dim)]
                isl_repr = f'{self.name}[{", ".join(map(str, args))}]'
                self.calc(tensor_table, *args)
                self.access = record_effective_op.get_tensor_access(isl_repr)
        return self.access


'''
N, M, Q = 512, 512, 1024
example_tensor_table = TensorTable()
example_tensor_table.add_tensor('A', [N, Q])
example_tensor_table.add_tensor('B', [Q, M])
example_tensor_table.add_tensor('C', [N, M])


def _example_statements():
    def S0(t, i, j):
        t['C'][i, j] = 0.0

    def S1(t, i, j, k):
        t['C'][i, j] = t['C'][i, j] + t['A'][i, k] * t['B'][k, j]

    return {
        'S0': Statement('S0', 2, S0),
        'S1': Statement('S1', 3, S1),
    }

example_iter_var_table = IterVarTable()
_tmp_vars = [example_iter_var_table.push() for _ in range(3)]

example_statements = _example_statements()

for statement in example_statements.values():
    statement.get_access(example_tensor_table)

for i in example_statements.values():
    j = i.to_tvm(example_tensor_table, *_tmp_vars[:i.dim])
    print(type(j), j)
    j = i.get_access(example_tensor_table)
    print(type(j), j)

for i in example_tensor_table.table.values():
    print(i.to_isl_set())
'''
