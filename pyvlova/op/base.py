from typing import Dict, Mapping, Iterable, Any, List, Callable

import tvm

from ..autotune.gpu_tile import tune_gpu_tile
from ..codegen.isl_to_tir import CUDANode2TIRParser, ISLNode2TIR, build_tvm_stmts
from ..poly.poly import TensorTable, Statement
from ..poly.schedule_tree import ScheduleTree
from ..utils import Mode

calc_mode = Mode()


class BaseOp(object):
    def __init__(self, schedule: ScheduleTree, tensors: TensorTable = None,
                 inputs: Iterable[str] = None, outputs: Iterable[str] = None,
                 statements: Mapping[str, Statement] = None):
        self.schedule: ScheduleTree = schedule
        self.tensors: TensorTable = tensors or TensorTable()
        self.inputs: List[str] = list(inputs or list())
        self.outputs: List[str] = list(outputs or list())
        self.statements: Dict[str, Statement] = dict(statements or dict())
        self._imp: Dict[str, Any] = dict()
        for s in self.statements.values():
            s.tensor_table = self.tensors

    def get_parser(self, factory=ISLNode2TIR):
        return factory(tensor_table=self.tensors, stmt_table=self.statements)

    def calc(self, *args, **kwargs):
        assert calc_mode.mode
        imp_name = 'calc_' + str(calc_mode.mode)
        if not hasattr(self, imp_name):
            return None
        return getattr(self, imp_name)(*args, **kwargs)

    def calc_tvm_llvm(self, *args, ctx=None, **kwargs):
        assert 'tvm_llvm' in self._imp
        assert len(args) <= len(self.inputs)

        if ctx is None:
            ctx = tvm.cpu()

        for i, arg in enumerate(args):
            kwargs[self.inputs[i]] = arg
        for i in self.inputs:
            assert i in kwargs
            assert isinstance(kwargs[i], tvm.nd.NDArray)
        for i in self.outputs:
            if i not in kwargs:
                kwargs[i] = tvm.nd.empty(self.tensors[i].shape, self.tensors[i].dtype, ctx=ctx)

        imp, arg_map = self._imp['tvm_llvm']
        imp_args = [None] * (len(arg_map))
        for k, i in arg_map.items():
            imp_args[i] = kwargs[k]
        imp(*imp_args)

        outputs = [imp_args[arg_map[k]] for k in self.outputs]
        if len(self.outputs) <= 0:
            return outputs[0]
        return tuple(outputs)

    def calc_tvm_cuda(self, *args, ctx=None, **kwargs):
        assert 'tvm_cuda' in self._imp
        assert len(args) <= len(self.inputs)

        if ctx is None:
            ctx = tvm.gpu()

        for i, arg in enumerate(args):
            kwargs[self.inputs[i]] = arg
        for i in self.inputs:
            assert i in kwargs
            assert isinstance(kwargs[i], tvm.nd.NDArray)
        for i in self.outputs:
            if i not in kwargs:
                kwargs[i] = tvm.nd.empty(self.tensors[i].shape, self.tensors[i].dtype, ctx=ctx)

        imp, arg_map = self._imp['tvm_cuda']
        imp_args = [None] * (len(arg_map))
        for k, i in arg_map.items():
            imp_args[i] = kwargs[k]
        imp(*imp_args)

        outputs = [imp_args[arg_map[k]] for k in self.outputs]
        if len(self.outputs) <= 0:
            return outputs[0]
        return tuple(outputs)

    def imp(self, *args, **kwargs):
        assert calc_mode.mode
        imp_name = 'imp_' + str(calc_mode.mode)
        if not hasattr(self, imp_name):
            return None
        return getattr(self, imp_name)(*args, **kwargs)

    def imp_tvm_llvm(self, te_tensors=None):
        if te_tensors is None:
            te_tensors = [i.te_tensor for i in self.tensors]
        for i in te_tensors:
            assert i.name in self.tensors
        name = f'{type(self).__name__}_{id(self)}'
        parser = self.get_parser(factory=ISLNode2TIR)
        tree = self.schedule.copy()
        stmts, tensors = build_tvm_stmts(name, tree, parser, te_tensors=te_tensors)
        assert all((i.name == j.name for i, j in zip(te_tensors, tensors)))
        with tvm.target.create('llvm'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(tensors)}
        self._imp['tvm_llvm'] = (func, arg_map)
        return func

    def imp_tvm_cuda(self, tile_size=None, te_tensors=None, gpu_tile_tune_kwargs=None):
        if te_tensors is None:
            te_tensors = [i.te_tensor for i in self.tensors]
        for i in te_tensors:
            assert i.name in self.tensors
        name = f'{type(self).__name__}_{id(self)}'
        parser = self.get_parser(factory=CUDANode2TIRParser)
        if tile_size is None:
            # noinspection PyTypeChecker
            tile_size, _ = tune_gpu_tile(name, self.schedule, parser, **(gpu_tile_tune_kwargs or {}))
        tree = self.schedule.copy()
        tree.gpu_tile(tile_size)
        stmts, tensors = build_tvm_stmts(name, tree, parser, te_tensors=te_tensors)
        assert all((i.name == j.name for i, j in zip(te_tensors, tensors)))
        with tvm.target.create('cuda'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(tensors)}
        self._imp['tvm_cuda'] = (func, arg_map)
        return func


class ArgumentedOp(BaseOp):
    required_params: Iterable[str] = []
    optional_params: Dict[str, Any] = {}
    calculated_params: Dict[str, Callable] = {}
    tensor_order: Iterable[str] = []
    inputs: List[str] = []
    outputs: List[str] = []
    schedule_factory: Callable = None
    tensors_factory: Callable = None
    statements_factory: Callable = None

    def __init__(self, **kwargs):
        self.params = dict()
        for i in self.required_params:
            assert i in kwargs, f'{i} not in {kwargs}'
            v = kwargs.pop(i)
            setattr(self, i, v)
            self.params[i] = v
        for i, d in self.optional_params.items():
            v = kwargs.pop(i, d)
            setattr(self, i, v)
            self.params[i] = v
        for k, f in self.calculated_params.items():
            v = kwargs.pop(k, f(**kwargs, **self.params))
            setattr(self, k, v)
            self.params[k] = v
        if 'tensor_order' in kwargs:
            self.tensor_order = kwargs.pop('tensor_order')
        for k in ('inputs', 'outputs'):
            if k not in kwargs:
                kwargs[k] = getattr(type(self), k)
        assert set(kwargs.keys()) == {'inputs', 'outputs'}
        super().__init__(
            schedule=type(self).schedule_factory(**self.params),
            tensors=type(self).tensors_factory(**self.params),
            statements=type(self).statements_factory(**self.params),
            **kwargs
        )

    def imp_tvm_llvm(self, te_tensors=None, **kwargs):
        if te_tensors is None:
            te_tensors = [self.tensors[i].te_tensor for i in self.tensor_order]
        return super().imp_tvm_llvm(te_tensors=te_tensors)

    def imp_tvm_cuda(self, te_tensors=None, **kwargs):
        if te_tensors is None:
            te_tensors = [self.tensors[i].te_tensor for i in self.tensor_order]
        return super().imp_tvm_cuda(te_tensors=te_tensors, **kwargs)
