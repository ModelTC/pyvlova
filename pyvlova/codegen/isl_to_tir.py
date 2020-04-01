import threading
from contextlib import contextmanager
from typing import Dict, Iterable
import tvm
from tvm import tir, te
import isl

from pyvlova.polyhedral.statement import IterVarTable, CUDAIterVarTable, TensorTable, Statement, Tensor


class Parser(object):
    def parse(self, *args, **kwargs):
        pass


class ISLNodeParser(Parser):
    def __init__(self, ast_build=None):
        super().__init__()
        self.ast_build = ast_build or isl.ast_build()
        for attr in dir(self.ast_build):
            if not attr.startswith('set_'):
                continue
            name = attr[len('set_'):]
            if hasattr(self, name):
                self.ast_build = getattr(self.ast_build, attr)(getattr(self, name))

    def parse(self, node, parent=None):
        if isinstance(node, isl.schedule):
            node = self.ast_build.node_from(node)
        assert type(node).__name__.startswith('ast_node_')
        t = type(node).__name__[len('ast_node_'):]
        assert t in ['block', 'for', 'if', 'list', 'mark', 'user']
        return getattr(self, f'parse_{t}')(node, parent)


class ISLExprParser(Parser):
    def parse(self, node, parent=None):
        assert type(node).__name__.startswith('ast_expr_')
        t = type(node).__name__[len('ast_expr_'):]
        return getattr(self, f'parse_{t}')(node, parent)


class ISLExpr2TIR(ISLExprParser):
    def __init__(self, iter_var_table):
        super().__init__()
        self.iter_var_table = iter_var_table

    def parse_int(self, expr, parent):
        return tir.IntImm('int32', int(expr.to_C_str()))

    def parse_id(self, expr, parent):
        return self.iter_var_table[expr.to_C_str()]

    def parse_op_mul(self, expr, parent):
        return tir.Mul(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_add(self, expr, parent):
        if expr.n_arg() == 1:
            return tir.Add(tir.IntImm('int32', 0), self.parse(expr.arg(0), expr))
        return tir.Add(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_minus(self, expr, parent):
        if expr.n_arg() == 1:
            return tir.Sub(tir.IntImm('int32', 0), self.parse(expr.arg(0), expr))
        return tir.Sub(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    parse_op_sub = parse_op_minus

    def parse_op_le(self, expr, parent):
        return tir.LE(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_lt(self, expr, parent):
        return tir.LT(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_ge(self, expr, parent):
        return tir.GE(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_gt(self, expr, parent):
        return tir.GT(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_and(self, expr, parent):
        return tir.And(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_or(self, expr, parent):
        return tir.Or(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_fdiv_q(self, expr, parent):
        return tir.FloorDiv(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_min(self, expr, parent):
        return tir.Min(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

    def parse_op_max(self, expr, parent):
        return tir.Max(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))


class ISLNode2TIR(ISLNodeParser):
    def __init__(self, tensor_table=None, stmt_table=None, iter_var_table=None, expr_parser=None, **kwargs):
        super().__init__(**kwargs)
        self.tensor_table = tensor_table or TensorTable()
        self.stmt_table: Dict[Statement] = stmt_table or dict()
        self.iter_var_table = iter_var_table or IterVarTable()
        self.expr_parser = expr_parser or ISLExpr2TIR(self.iter_var_table)
        self.mark_stack = []

    def parse_mark(self, node, parent):
        mark = node.id().name()
        self.mark_stack.append(mark)
        res = self.parse(node.node(), node)
        self.mark_stack.pop()
        return res

    @contextmanager
    def _for_loop_util(self, node):
        if self.mark_stack and self.mark_stack[-1].isdigit():
            for_type = int(self.mark_stack[-1])
        else:
            for_type = 0

        iter_var = self.iter_var_table.push()
        c_var = self.iter_var_table.push(node.iterator().to_C_str())
        extent_var = self.iter_var_table.push('_isl_' + node.iterator().to_C_str() + '_extent')

        assert node.cond().arg(0).to_C_str() == node.iterator().to_C_str()

        lower = self.expr_parser.parse(node.init())
        upper = self.expr_parser.parse(node.cond().arg(1))
        step = self.expr_parser.parse(node.inc())
        if isinstance(node.cond(), (isl.ast_expr_op_le, isl.ast_expr_op_ge)):
            upper = tir.Add(upper, step)

        yield iter_var, c_var, extent_var, lower, upper, step, for_type

        assert self.iter_var_table.pop() is extent_var
        assert self.iter_var_table.pop() is c_var
        assert self.iter_var_table.pop() is iter_var

    def parse_for(self, node, parent):
        with self._for_loop_util(node) as (iter_var, c_var, extent_var, lower, upper, step, for_type):
            extent = tir.FloorDiv(tir.Sub(upper, lower), step)
            return tir.LetStmt(
                extent_var, extent,
                tir.For(
                    iter_var, tir.IntImm('int32', 0), extent_var, for_type, 0,
                    tir.LetStmt(
                        c_var, tir.Add(tir.Mul(iter_var, step), lower),
                        self.parse(node.body(), node)
                    )
                )
            )

    def parse_user(self, node, parent):
        assert isinstance(node.expr(), isl.ast_expr_op_call)
        e = node.expr()
        sid = e.arg(0).to_C_str()
        if sid.startswith('_'):
            return None
        args = [self.expr_parser.parse(e.arg(i)) for i in range(1, e.n_arg())]
        return self.stmt_table[sid].to_tvm(self.tensor_table, *args)

    def parse_block(self, node, parent):
        children = node.children()
        body = []
        for i in range(children.size()):
            child = self.parse(children.at(i), parent)
            if child is not None:
                body.append(child)
        return tir.SeqStmt(body)

    def parse_if(self, node, parent):
        cond = self.expr_parser.parse(node.cond())
        then_node = self.parse(node.then_node(), node)
        if node.has_else_node():
            else_node = self.parse(node.else_node(), node)
        else:
            else_node = None
        return tir.IfThenElse(condition=cond, then_case=then_node, else_case=else_node)


class CUDANode2TIRParser(ISLNode2TIR):
    def __init__(self, cuda_iter_var_table=None, **kwargs):
        super().__init__(**kwargs)
        self.cuda_iter_var_table = cuda_iter_var_table or CUDAIterVarTable()

    def _produce_tensors(self, scope, tensors: Iterable[Tensor], body):
        for t in reversed(list(tensors)):
            body = t.build_tir_realize(scope, body)
        return body

    def parse(self, node, parent=None):
        if parent is None:
            body = super().parse(node, parent)
            write_tensors = set()
            for i in self.stmt_table.values():
                 write_tensors = write_tensors.union(i.get_access()[1]['write'])
            return self._produce_tensors('', write_tensors, body)
        return super().parse(node, parent)

    def before_each_mark(self, node, ast_build):
        # TODO: calculate tensor offset in kernel
        # name = node.name()
        # if name in self.tensor_table:
        #     for t in self.tensor_table[name]:
        #         e = offset_to_ast(t[1], ast_build)
        #         t[1] = [e.arg(i) for i in range(1, e.n_arg())]
        pass

    def parse_mark(self, node, parent):
        body = super().parse_mark(node, parent)
        # TODO: build local / shared memory realize node
        # name = node.id().name()
        # if name in self.tensor_table:
        #     return self._produce_tensors(name, body)
        return body

    def parse_for(self, node, parent):
        if self.mark_stack and self.mark_stack[-1].startswith('bind='):
            _, axis = self.mark_stack[-1].rsplit('=', 1)
            with self._for_loop_util(node) as (_, c_var, extent_var, lower, upper, step, __):
                with self.cuda_iter_var_table.var(axis) as iter_var:
                    extent = tir.FloorDiv(tir.Sub(upper, lower), step)
                    return tir.LetStmt(
                        extent_var, extent,
                        tir.AttrStmt(
                            node=iter_var, attr_key='thread_extent', value=extent_var,
                            body=tir.LetStmt(
                                c_var, tir.Add(tir.Mul(iter_var.var, step), lower),
                                self.parse(node.body(), node)
                            )
                        )
                    )
        return super().parse_for(node, parent)


def build_tvm_stmts(name, tree, parser: ISLNode2TIR):
    if isinstance(tree, ScheduleTree):
        tree = tree.to_isl()
    stmts = parser.parse(tree)
    stmts = tir.ir_pass.CanonicalSimplify(stmts)

    tensor_table = parser.tensor_table
    tensors = [i.te_tensor for i in tensor_table.table.values()]
    binds, arg_list = tvm.driver.build_module.get_binds(tensors)

    with threading.Lock():
        old_form_body = tvm.driver.build_module.form_body
        old_get_binds = tvm.driver.build_module.get_binds
        try:
            tvm.driver.build_module.form_body = lambda x: stmts
            tvm.driver.build_module.get_binds = lambda *_, **__: (binds, arg_list)

            tvm_s = te.create_schedule(tensors[-1].op)
            stmts = tvm.lower(tvm_s, tensors, name=name)
        except:
            raise
        finally:
            tvm.driver.build_module.get_binds = old_get_binds
            tvm.driver.build_module.form_body = old_form_body

    return stmts


from ..polyhedral.statement import example_tensor_table, example_statements, N, M, Q
from ..polyhedral.schedule_tree import example_tree, ScheduleTree

parser = CUDANode2TIRParser(
    tensor_table=example_tensor_table,
    stmt_table=example_statements
)

'''
tree = example_tree.copy()
tree.apply_params(n=N, m=M, q=Q)
tree.gpu_tile([7, 7])

print(tree.to_yaml())

stmts = build_tvm_stmts('example_to_tvm', tree, parser)
print(stmts.body)

ctx = tvm.gpu()
import numpy as np

a_np = np.random.uniform(size=example_tensor_table['A'].shape).astype(example_tensor_table['A'].dtype)
b_np = np.random.uniform(size=example_tensor_table['B'].shape).astype(example_tensor_table['B'].dtype)
c_np = a_np @ b_np
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.empty(c_np.shape, ctx=ctx)

with tvm.target.create('cuda'):
    func = tvm.build(stmts, name='example_to_tvm')

print(func.imported_modules[0].get_source(), end='\n\n')

func(a, b, c)

tvm.testing.assert_allclose(c_np, c.asnumpy(), rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('polyhedral speed:', evaluator(a, b, c).mean, end='\n\n')
'''
