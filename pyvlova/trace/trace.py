import sys
import contextlib


def wrap_sys_trace():
    old_sys_trace = sys.gettrace() or (lambda *_, **__: None)

    def sys_trace(*args, **kwargs):
        # TODO: monitor if statements and calls
        return old_sys_trace(*args, **kwargs)

    sys.settrace(sys_trace)
    return old_sys_trace


class CurrentStatus(object):
    def __init__(self):
        self.tracing = False
        self.monkey_patched = False
        self.iter_vars = []
        self.iter_vars_s = set()
        self.constraints = []

    def push_iter_var(self, name=None):
        if name is None:
            name = f'v{len(self.iter_vars)}'
        assert name not in self.iter_vars_s
        self.iter_vars.append(name)
        self.iter_vars_s.add(name)
        print('Pushed an iteration variable', name)
        return name

    def pop_iter_var(self):
        name = self.iter_vars.pop()
        self.iter_vars_s.remove(name)
        print('Popped the iteration variable', name)
        return name

    def push_constraint(self, c):
        print('Pushed a constraint', c)
        self.constraints.append(c)

    def pop_constraint(self):
        print('Popped the constraint', self.constraints[-1])
        return self.constraints.pop()

    @contextlib.contextmanager
    def start_tracing(self):
        self.tracing = True
        old_sys_trace = wrap_sys_trace()
        try:
            from . import patch
            with patch.monkey_patched():
                yield self
        finally:
            self.tracing = False
            sys.settrace(old_sys_trace)


current_status = CurrentStatus()
