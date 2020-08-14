import functools
import sympy


class ISLReprPrinter(sympy.StrPrinter):
    def _print_And(self, expr):
        return self.stringify(expr.args, ' and ', sympy.printing.str.PRECEDENCE['And'])

    def _print_Or(self, expr):
        return self.stringify(expr.args, ' or ', sympy.printing.str.PRECEDENCE['Or'])

    def _print_Function(self, expr):
        if expr.func is sympy.Mod:
            r = self.stringify(expr.args, ' mod ', sympy.printing.str.PRECEDENCE['Mul'] + 1)
            return f'{r}'
        return super()._print_Function(expr)

    def _print_Relational(self, expr):
        if isinstance(expr, sympy.Eq):
            return self.stringify(expr.args, ' = ', sympy.printing.str.PRECEDENCE['Relational'])
        return super()._print_Relational(expr)


def parse_sympy_to_isl_repr(expr):
    return ISLReprPrinter().doprint(expr)


def constraints_to_isl_repr(constraints):
    return parse_sympy_to_isl_repr(functools.reduce(sympy.And, constraints))
