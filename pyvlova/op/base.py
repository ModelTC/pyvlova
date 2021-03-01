# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from itertools import chain
from typing import Dict, Iterable, Any, List, Callable
import os

import numpy
import tvm
from tvm import autotvm

from ..autotune import tune_cuda_tile, default_tune_eval_settings, default_timing_eval_settings
from ..codegen import CUDAISLNode2TIR, ISLNode2TIR, lower_tvm_stmt
from ..poly import Tensor, Statement, TensorTable, ScheduleTree, cuda_tile, cuda_find_sharable_tensors
from ..utils import Mode, filter_contains, slugify


calc_mode = Mode()


class OpParameter(object):
    def __init__(self, name):
        self.name = name

    @property
    def hidden_attr(self):
        return '__op_parameter_' + self.name

    def mock_poly_tensor(self, instance, tensor: Tensor):
        setattr(instance, self.hidden_attr, numpy.random.random(tensor.shape).astype(tensor.dtype))

    def __get__(self, instance, owner):
        value = getattr(instance, self.hidden_attr, None)
        if 'tvm' in calc_mode.mode:
            if 'cuda' in calc_mode.mode:
                return tvm.nd.array(value, ctx=tvm.gpu())
            return tvm.nd.array(value)
        return value

    def __set__(self, instance, value):
        if not calc_mode.mode:
            if isinstance(value, Tensor):
                self.mock_poly_tensor(instance, value)
            else:
                setattr(instance, self.hidden_attr, value)
            return
        if 'tvm' in calc_mode.mode:
            if isinstance(value, tvm.nd.NDArray):
                value = value.asnumpy()
        getattr(instance, self.hidden_attr)[:] = value


class BaseOp(object):
    def __init__(self, name: str = ''):
        self._imp: Dict[str, Any] = dict()
        self.name = name or f'{type(self).__name__}_{id(self)}'

    def __getattr__(self, key):
        raise AttributeError(f'no such attribute {key} for Op {type(self).__name__}')

    def calc(self, *args, **kwargs):
        raise NotImplemented

    def imp(self, *args, **kwargs):
        raise NotImplemented


class CombinedOp(BaseOp):
    def __init__(self, ops=None, name=''):
        super().__init__(name=name)
        self._ops = ops or []

    def imp(self, *args, **kwargs):
        return [op.imp(*args, **kwargs) for op in self._ops]


class SequenceOp(CombinedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._ops:
            op = self._ops[-1]
            for i in ['batch', 'channel', 'height', 'width', 'feature']:
                v = None
                if hasattr(op, i):
                    v = getattr(op, i)
                elif hasattr(op, 'out_' + i):
                    v = getattr(op, 'out_' + i)
                if v is not None:
                    setattr(self, 'out_' + i, v)

    def calc(self, *args, **kwargs):
        res = self._ops[0].calc(*args, **kwargs)
        for op in self._ops[1:]:
            if not isinstance(res, tuple):
                res = (res,)
            res = op.calc(*res)
        return res


class PolyOp(BaseOp):
    def __init__(self, schedule: ScheduleTree, tensors: TensorTable = None,
                 inputs: Iterable[str] = None, outputs: Iterable[str] = None,
                 statements=None, name: str = ''):
        super().__init__(name)
        self.schedule: ScheduleTree = schedule
        self.tensors: TensorTable = tensors or TensorTable()
        self.inputs: List[str] = list(inputs or list())
        self.outputs: List[str] = list(outputs or list())
        if statements is None:
            statements = dict()
        if isinstance(statements, List):
            statements = {f.__name__: Statement.from_calc(f) for f in statements}
        self.statements: Dict[str, Statement] = dict(statements)
        for s in self.statements.values():
            s.tensor_table = self.tensors

    def get_parser(self, factory=ISLNode2TIR, **kwargs):
        return factory(tensor_table=self.tensors, stmt_table=self.statements, **kwargs)

    def calc(self, *args, **kwargs):
        assert calc_mode.mode
        imp_name = 'calc_' + str(calc_mode.mode)
        if not hasattr(self, imp_name):
            raise Exception('no such implement, ' + str(calc_mode.mode))
        return getattr(self, imp_name)(*args, **kwargs)

    def imp(self, *args, **kwargs):
        assert calc_mode.mode
        imp_name = 'imp_' + str(calc_mode.mode)
        if not hasattr(self, imp_name):
            raise Exception('no such implement, ' + str(calc_mode.mode))
        return getattr(self, imp_name)(*args, **kwargs)


class PolyTVMOp(PolyOp):
    def _calc_on_tvm(self, *args, ctx=None, _imp_name=None, **kwargs):
        assert len(args) <= len(self.inputs)
        assert ctx and _imp_name
        if _imp_name not in self._imp:
            getattr(self, 'imp_' + _imp_name)()

        for i, arg in enumerate(args):
            kwargs[self.inputs[i]] = arg
        for i in self.inputs:
            assert i in kwargs
            assert isinstance(kwargs[i], tvm.nd.NDArray)
        for i in self.outputs:
            if i not in kwargs:
                kwargs[i] = tvm.nd.empty(self.tensors[i].shape, self.tensors[i].dtype, ctx=ctx)

        imp, arg_map = self._imp[_imp_name]
        imp_args = [None] * len(arg_map)
        for k, i in arg_map.items():
            imp_args[i] = kwargs[k]
        imp(*imp_args)

        outputs = [imp_args[arg_map[k]] for k in self.outputs]
        if len(self.outputs) <= 1:
            return outputs[0]
        return tuple(outputs)

    def imp_tvm_llvm(self, te_tensors=None):
        if te_tensors is None:
            te_tensors = [i.te_tensor for i in self.tensors]
        for i in te_tensors:
            assert i.name in self.tensors
        name = self.name + '_tvm_llvm'
        parser = self.get_parser(factory=ISLNode2TIR)
        tree = self.schedule.copy()
        stmts, tensors = build_tvm_stmts(name, tree, parser, te_tensors=te_tensors)
        assert all((i.name == j.name for i, j in zip(te_tensors, tensors)))
        with tvm.target.Target('llvm'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(tensors)}
        self._imp['tvm_llvm'] = (func, arg_map)
        return func

    def imp_tvm_cuda(self, tile_size=None, te_tensors=None, do_shared_opt=True, tune_kwargs=None):
        if te_tensors is None:
            te_tensors = [i.te_tensor for i in self.tensors]
        for i in te_tensors:
            assert i.name in self.tensors
        name = self.name + '_tvm_cuda'
        parser = self.get_parser(factory=CUDAISLNode2TIR, do_shared_opt=do_shared_opt)
        if tile_size is None:
            tile_size, _ = tune_cuda_tile(
                name, self.schedule, te_tensors, parser, **(tune_kwargs or {}))
        tree = self.schedule.copy()
        cuda_tile(tree, tile_size)
        stmt = parser.parse(tree)
        stmts = lower_tvm_stmt(stmt, te_tensors, name=name)
        with tvm.target.Target('cuda'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(te_tensors)}
        self._imp['tvm_cuda'] = (func, arg_map)
        return func

    def __getattr__(self, key):
        if key.startswith('calc_tvm_'):
            name = key[len('calc_'):]
            if 'cuda' in name or 'nvptx' in name or 'gpu' in name:
                default_ctx = tvm.gpu()
            else:
                default_ctx = tvm.cpu()

            def calc(*args, ctx=None, **kwargs):
                return self._calc_on_tvm(*args, ctx=ctx or default_ctx, _imp_name=name, **kwargs)

            setattr(self, key, calc)
            return calc

        if key.startswith('imp_tvm_') and key.endswith('_timing'):
            target = key[len('imp_tvm_'):-len('_timing')]

            def _imp_tvm_timing(*args, timing_number=20, **kwargs):
                if 'tvm_' + target not in self._imp:
                    getattr(self, 'imp_tvm_' + target)(*args, **kwargs)
                func, arg_map = self._imp['tvm_' + target]

                def timing(*t_args, **t_kwargs):
                    ctx = None
                    for i in chain(t_args, t_kwargs.values()):
                        if isinstance(i, tvm.nd.NDArray):
                            ctx = i.ctx
                            break
                    evaluator = func.time_evaluator(func.entry_name, ctx, **default_timing_eval_settings)
                    t = evaluator(*t_args, **t_kwargs).mean
                    print(self.name, 'tvm timing', target, '%.9f us' % (t * 1e6))
                    timing.t = t

                self._imp[f'tvm_{target}_timing'] = (timing, arg_map)

            setattr(self, key, _imp_tvm_timing)
            return _imp_tvm_timing

        super().__getattr__(key)

    topi_cuda_task_name: str = ''

    def topi_cuda_args(self, **_):
        return []

    topi_cuda_calc_func: Callable = None
    topi_cuda_schedule_func: Callable = None
    topi_cuda_calc_ret_map: List[str] = []

    def _tune_topi_cuda(self, name, args, te_tensors, tune_kwargs):
        n_trial = tune_kwargs.get('n_trial', 40)
        preserve_log = tune_kwargs.get('preserve_log', False)
        tmp_file_name = slugify(name) + '.topi_cuda.log'
        if n_trial > 0:
            task = autotvm.task.create(self.topi_cuda_task_name, args=args, target='cuda')
            tuner = tune_kwargs.get('tuner', autotvm.tuner.XGBTuner(task))
            tuner.tune(
                n_trial=n_trial,
                measure_option={
                    'builder': tune_kwargs.get('builder', autotvm.LocalBuilder()),
                    'runner': tune_kwargs.get('runner', autotvm.LocalRunner(timeout=20, **default_tune_eval_settings)),
                },
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=f'TOPI {name}'),
                    autotvm.callback.log_to_file(tmp_file_name),
                    *tune_kwargs.get('callbacks', [])
                ]
            )
        with autotvm.apply_history_best(tmp_file_name):
            result = self._build_topi_cuda(name, args, te_tensors)
        if not preserve_log:
            os.remove(tmp_file_name)
        return result

    def _build_topi_cuda(self, name, args, te_tensors):
        res = type(self).topi_cuda_calc_func(*args)
        if isinstance(res, tvm.te.Tensor):
            res = (res,)
        named_res = dict(zip(self.topi_cuda_calc_ret_map, res))
        for i in range(len(te_tensors)):
            if te_tensors[i].name in named_res:
                te_tensors[i] = named_res[te_tensors[i].name]
        s = type(self).topi_cuda_schedule_func(res)
        func = tvm.build(s, te_tensors, name=slugify(name))
        return func

    def imp_tvm_topi_cuda(self, te_tensors=None, tune_kwargs=None):
        if tune_kwargs is None:
            tune_kwargs = {}
        assert te_tensors
        for i in te_tensors:
            assert i.name in self.tensors
        ts = {i.name: i for i in te_tensors}
        name = self.name + '_tvm_topi_cuda'
        with tvm.target.Target('cuda'):
            args = self.topi_cuda_args(**ts)
            if self.topi_cuda_task_name:
                func = self._tune_topi_cuda(name, args, te_tensors, tune_kwargs)
            else:
                func = self._build_topi_cuda(name, args, te_tensors)
        arg_map = {v: i for i, v in enumerate(self.tensor_order)}
        self._imp['tvm_topi_cuda'] = (func, arg_map)
        return func


class ArgumentedOp(PolyTVMOp):
    required_args: Iterable[str] = []
    optional_args: Dict[str, Any] = {}
    calculated_args: Dict[str, Callable] = {}
    tensor_order: Iterable[str] = []
    inputs: List[str] = []
    outputs: List[str] = []
    schedule_factory: Callable = None
    tensors_factory: Callable = None
    statements_factory: Callable = None

    @classmethod
    def filter_args(cls, args):
        return filter_contains(args, cls.required_args, cls.optional_args, cls.calculated_args)

    def __init__(self, **kwargs):
        self.arguments = dict()
        for i in self.required_args:
            assert i in kwargs, f'{i} not in {kwargs}'
            v = kwargs.pop(i)
            setattr(self, i, v)
            self.arguments[i] = v
        for i, d in self.optional_args.items():
            v = kwargs.pop(i, d)
            setattr(self, i, v)
            self.arguments[i] = v
        for k, f in self.calculated_args.items():
            v = kwargs.pop(k, f(**kwargs, **self.arguments))
            setattr(self, k, v)
            self.arguments[k] = v
        if 'tensor_order' in kwargs:
            self.tensor_order = kwargs.pop('tensor_order')
        for k in ('inputs', 'outputs'):
            if k not in kwargs:
                kwargs[k] = getattr(type(self), k)
        assert set(kwargs.keys()).issubset({'inputs', 'outputs', 'name'})
        super().__init__(
            schedule=type(self).schedule_factory(**self.arguments),
            tensors=type(self).tensors_factory(**self.arguments),
            statements=type(self).statements_factory(**self.arguments),
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

    def imp_tvm_topi_cuda(self, te_tensors=None, **kwargs):
        if te_tensors is None:
            te_tensors = [self.tensors[i].te_tensor for i in self.tensor_order]
        super().imp_tvm_topi_cuda(te_tensors=te_tensors, **kwargs)
