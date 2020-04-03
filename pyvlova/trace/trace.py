import sys
import contextlib
import collections
import loopy as lp

from ..codegen import sympy2isl


def wrap_sys_trace():
    old_sys_trace = sys.gettrace() or (lambda *_, **__: None)

    def sys_trace(frame, arg):
        # TODO: monitor if statements and calls
        return old_sys_trace(frame, arg)

    sys.settrace(sys_trace)
    return old_sys_trace


class CurrentStatus(object):
    def __init__(self):
        self.tracing = False
        self.monkey_patched = False
        self.known_tensors = []
        self.known_tensors_s = collections.defaultdict(list)
        self.iter_vars = []
        self.iter_vars_s = set()
        self.constraints = []
        self.statements = []
        self.loopy_kernels = []

    def pop_statements(self):
        if not self.statements:
            return

        vars_isl_repr = '[%s]' % ', '.join(map(str, self.iter_vars))
        cons_isl_repr = sympy2isl.constraints_to_isl_repr(self.constraints)
        domain = f'{{ {vars_isl_repr}: {cons_isl_repr} }}'
        statement = '\n'.join(map(lambda x: x.to_loopy(), self.statements))

        print(domain)
        print(statement)
        kernel = lp.make_kernel(domain, statement)
        kernel = lp.add_dtypes(kernel, {k: v[-1].dtype for k, v in self.known_tensors_s.items()})

        self.loopy_kernels.append(kernel)
        self.statements = []

    def push_iter_var(self, name=None):
        if name is None:
            name = f'v{len(self.iter_vars)}'
        assert name not in self.iter_vars_s
        self.iter_vars.append(name)
        self.iter_vars_s.add(name)
        print('Pushed an iteration variable', name)
        return name

    def pop_iter_var(self):
        self.pop_statements()
        name = self.iter_vars.pop()
        self.iter_vars_s.remove(name)
        print('Popped the iteration variable', name)
        return name

    def push_tensor(self, tensor):
        self.known_tensors.append(tensor)
        self.known_tensors_s[str(tensor.symbol)].append(tensor)
        print('Pushed a tensor', str(tensor.symbol))
        return tensor

    def pop_tensor(self):
        self.pop_statements()
        tensor = self.known_tensors.pop()
        key = str(tensor.symbol)
        self.known_tensors_s[key].pop()
        if not self.known_tensors_s[key]:
            del self.known_tensors_s[key]
        print('Pop the tensor', str(tensor.symbol))
        return tensor

    def push_constraint(self, c):
        print('Pushed a constraint', c)
        self.constraints.append(c)

    def pop_constraint(self):
        self.pop_statements()
        print('Popped the constraint', self.constraints[-1])
        return self.constraints.pop()

    def add_statement(self, statement):
        self.statements.append(statement)

    @contextlib.contextmanager
    def start_tracing(self):
        self.tracing = True
        old_sys_trace = wrap_sys_trace()
        try:
            from . import patch
            with patch.monkey_patched():
                self.statements = []
                self.loopy_kernels = []
                yield self
        finally:
            self.tracing = False
            sys.settrace(old_sys_trace)


current_status = CurrentStatus()
