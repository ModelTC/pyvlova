from itertools import chain
from typing import Dict, Mapping, Iterable, Any, List, Callable

import tvm
from tvm import autotvm

from ..autotune.gpu_tile import tune_gpu_tile
from ..codegen.isl_to_tir import CUDANode2TIRParser, ISLNode2TIR, build_tvm_stmts
from ..poly.poly import TensorTable, Statement
from ..poly.schedule_tree import ScheduleTree
from ..utils import Mode

calc_mode = Mode()


class BaseOp(object):
    def __init__(self, schedule: ScheduleTree, tensors: TensorTable = None,
                 inputs: Iterable[str] = None, outputs: Iterable[str] = None,
                 statements: Mapping[str, Statement] = None, name: str = ''):
        self.schedule: ScheduleTree = schedule
        self.tensors: TensorTable = tensors or TensorTable()
        self.inputs: List[str] = list(inputs or list())
        self.outputs: List[str] = list(outputs or list())
        self.statements: Dict[str, Statement] = dict(statements or dict())
        self._imp: Dict[str, Any] = dict()
        self.name = name or f'{type(self).__name__}_{id(self)}'
        for s in self.statements.values():
            s.tensor_table = self.tensors

    def get_parser(self, factory=ISLNode2TIR):
        return factory(tensor_table=self.tensors, stmt_table=self.statements)

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
        with tvm.target.create('llvm'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(tensors)}
        self._imp['tvm_llvm'] = (func, arg_map)
        return func

    def imp_tvm_cuda(self, tile_size=None, te_tensors=None, tune_kwargs=None):
        if te_tensors is None:
            te_tensors = [i.te_tensor for i in self.tensors]
        for i in te_tensors:
            assert i.name in self.tensors
        name = self.name + '_tvm_cuda'
        parser = self.get_parser(factory=CUDANode2TIRParser)
        if tile_size is None:
            # noinspection PyTypeChecker
            tile_size, _ = tune_gpu_tile(name, self.schedule, parser, **(tune_kwargs or {}))
        tree = self.schedule.copy()
        tree.gpu_tile(tile_size)
        stmts, tensors = build_tvm_stmts(name, tree, parser, te_tensors=te_tensors)
        assert all((i.name == j.name for i, j in zip(te_tensors, tensors)))
        with tvm.target.create('cuda'):
            func = tvm.build(stmts, name=name)
        arg_map = {v.name: i for i, v in enumerate(tensors)}
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
                    evaluator = func.time_evaluator(func.entry_name, ctx, number=timing_number)
                    print(self.name, 'tvm timing', target, evaluator(*t_args, **t_kwargs).mean)
                self._imp[f'tvm_{target}_timing'] = (timing, arg_map)

            setattr(self, key, _imp_tvm_timing)
            return _imp_tvm_timing

        raise AttributeError


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
    topi_cuda_task_name: str = ''
    topi_cuda_args: Callable = None
    topi_cuda_calc_func: Callable = None
    topi_cuda_schedule_func: Callable = None
    topi_cuda_calc_ret_map: List[str] = []

    def imp_tvm_topi_cuda(self, te_tensors=None, tune_kwargs=None):
        assert self.topi_cuda_task_name
        if tune_kwargs is None:
            tune_kwargs = {}
        if te_tensors is None:
            te_tensors = [self.tensors[i].te_tensor for i in self.tensor_order]
        for i in te_tensors:
            assert i.name in self.tensors
        ts = {i.name: i for i in te_tensors}
        name = self.name + '_tvm_topi_cuda'
        with tvm.target.create('cuda'):
            args = type(self).topi_cuda_args(**ts)
            task = autotvm.task.create(self.topi_cuda_task_name, args=args, target='cuda')
            tmp_file_name = f'{name}.topi.log'
            tuner = tune_kwargs.get('tuner', autotvm.tuner.XGBTuner(task))
            n_trial = tune_kwargs.get('n_trial', 40)
            tuner.tune(
                n_trial=n_trial,
                measure_option={
                    'builder': tune_kwargs.get('builder', autotvm.LocalBuilder()),
                    'runner': tune_kwargs.get(
                        'runner', autotvm.LocalRunner(number=6, min_repeat_ms=100, timeout=20)),
                },
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=f'TOPI {name}'),
                    autotvm.callback.log_to_file(tmp_file_name),
                    *tune_kwargs.get('callbacks', [])
                ]
            )
            with autotvm.apply_history_best(tmp_file_name):
                with tvm.target.create('cuda'):
                    res = type(self).topi_cuda_calc_func(*args)
                    if isinstance(res, tvm.nd.NDArray):
                        res = (res, )
                    for t, name in zip(res, self.topi_cuda_calc_ret_map):
                        te_tensors[ts[name]] = t
                    s = type(self).topi_cuda_schedule_func(*args)
                    func = tvm.build(s, te_tensors, name=name)
        arg_map = {v.name: i for i, v in enumerate(te_tensors)}
        self._imp['tvm_topi_cuda'] = (func, arg_map)
        return func

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
        assert set(kwargs.keys()).issubset({'inputs', 'outputs', 'name'})
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
