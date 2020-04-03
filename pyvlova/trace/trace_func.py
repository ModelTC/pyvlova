import inspect

from .trace import current_status
from ..tensor.dummy_tensor import DummyTensor


class TracableFuncMeta(type):
    instances = dict()

    def __call__(cls, func):
        key = id(func)
        if key in cls.instances:
            return cls.instances[key]
        cls.instances[key] = super().__call__(func)
        return cls.instances[key]


class TracableCalcFunc(metaclass=TracableFuncMeta):
    def __init__(self, func):
        self.plain_func = func
        self.plain_src = inspect.getsource(func)
        # TODO: flatten source and add more info to source
        # self.func_ast = astor.code_to_ast(self.plain_src)
        # self.pretty_src = astor.to_source(self.func_ast)
        # self.pretty_func = utils.run_code(
        #     self.pretty_src, func.__globals__, utils.func_name(func)
        # )
        self.pretty_func = self.plain_func

    def __call__(self, *args, **kwargs):
        if current_status.tracing:
            print('Tracing', self.pretty_func.__name__)
            for arg in args:
                if isinstance(arg, DummyTensor) and not arg.wrapped:
                    current_status.push_tensor(arg)
            res = self.pretty_func(*args, **kwargs)
            for arg in args[::-1]:
                if isinstance(arg, DummyTensor) and not arg.wrapped:
                    current_status.pop_tensor()
            return res
        else:
            return self.plain_func(*args, **kwargs)

    def trace(self, *args, **kwargs):
        with current_status.start_tracing():
            return self(*args, **kwargs)
