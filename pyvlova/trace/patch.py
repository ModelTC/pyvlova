import contextlib
import sympy

from .trace import current_status
from .trace_func import TracableCalcFunc


patches = []


def register_patch(obj, name, new_val):
    patches.append((obj, name, new_val))


def replace_item(d, key, value):
    if key in d:
        old_value = d[key]

        def _fallback():
            d[key] = old_value
    else:
        def _fallback():
            del d[key]

    d[key] = value
    return _fallback


def monkey_patch():
    if current_status.monkey_patched is not False:
        return current_status.monkey_patched

    fallback_stack = []

    for obj, name, new_val in patches:
        if isinstance(obj, dict):
            fallback = replace_item(obj, name, new_val)
            fallback_stack.append(fallback)
        elif obj is not None:
            fallback = replace_item(obj.__dict__, name, new_val)
            fallback_stack.append(fallback)
        else:
            for func in TracableCalcFunc._instances.values():
                fallback = replace_item(
                    func.pretty_func.__globals__, name, new_val
                )
                fallback_stack.append(fallback)

    def recover():
        while fallback_stack:
            fallback = fallback_stack.pop()
            fallback()
        current_status.monkey_patched = False

    current_status.monkey_patched = recover
    return recover


@contextlib.contextmanager
def monkey_patched():
    if current_status.monkey_patched is False:
        recover = monkey_patch()
        try:
            yield
        finally:
            recover()
    else:
        yield


def when_tracing(fallback):
    def _when_tracing(func):
        def wrapped(*args, **kwargs):
            if current_status.tracing:
                return func(*args, **kwargs)
            return fallback(*args, **kwargs)
        return wrapped
    return _when_tracing


@when_tracing(range)
def my_range(left, right=None, step=None):
    if right is None:
        left, right, step = 0, left, 1
    elif step is None:
        step = 1

    assert isinstance(left, (int, sympy.Expr))
    assert isinstance(right, (int, sympy.Expr))
    assert isinstance(step, (int, sympy.Expr))
    assert not isinstance(step, int) or step != 0
    if isinstance(left, int) and isinstance(right, int) and left >= right:
        return

    var_name = current_status.push_iter_var()
    var = sympy.var(var_name)

    if step == 1:
        constraint = sympy.And(left <= var, var < right)
    else:
        constraint = sympy.And(left <= var, var < right,
            sympy.Eq((var - left) % step, 0))

    current_status.push_constraint(constraint)
    yield var
    current_status.pop_constraint()
    current_status.pop_iter_var()


register_patch(None, 'range', my_range)
