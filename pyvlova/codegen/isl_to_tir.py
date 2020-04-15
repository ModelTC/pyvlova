import threading
from contextlib import contextmanager
from functools import reduce
from typing import Dict, Iterable

import tvm
from tvm import tir, te
import isl

from pyvlova.poly.gpu import gpu_find_sharable_tensors
from pyvlova.poly.poly import IterVarTable, CUDAIterVarTable, TensorTable, Statement, Tensor
from pyvlova.poly.schedule_tree.tree import ScheduleTree
from pyvlova.utils import tir_imm, slugify, tir_sync


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
        if isinstance(node, ScheduleTree):
            node = node.to_isl()
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


# noinspection PyMethodMayBeStatic, PyUnusedLocal
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

    def parse_op_eq(self, expr, parent):
        return tir.EQ(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

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


# noinspection PyUnusedLocal
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

    def _for_bounds(self, node):
        assert node.cond().arg(0).to_C_str() == node.iterator().to_C_str()
        var_name = node.iterator().to_C_str()
        lower = self.expr_parser.parse(node.init())
        upper = self.expr_parser.parse(node.cond().arg(1))
        step = self.expr_parser.parse(node.inc())
        if isinstance(node.cond(), (isl.ast_expr_op_le, isl.ast_expr_op_ge)):
            upper = tir.Add(upper, step)

        return var_name, lower, upper, step

    @contextmanager
    def _for_loop_vars(self, node):
        if self.mark_stack and self.mark_stack[-1].isdigit():
            for_type = int(self.mark_stack[-1])
        else:
            for_type = 0

        c_var_name, lower, upper, step = self._for_bounds(node)

        iter_var = self.iter_var_table.push()
        c_var = self.iter_var_table.push(c_var_name)
        extent_var = self.iter_var_table.push('_isl_' + node.iterator().to_C_str() + '_extent')

        yield iter_var, c_var, extent_var, lower, upper, step, for_type

        assert self.iter_var_table.pop() is extent_var
        assert self.iter_var_table.pop() is c_var
        assert self.iter_var_table.pop() is iter_var

    def parse_for(self, node, parent):
        with self._for_loop_vars(node) as (iter_var, c_var, extent_var, lower, upper, step, for_type):
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
    def __init__(self, cuda_iter_var_table=None, shared_tensors=None, has_side_effect=False, **kwargs):
        super().__init__(**kwargs)
        self.cuda_iter_var_table: CUDAIterVarTable = cuda_iter_var_table or CUDAIterVarTable()
        self.shared_tensors = shared_tensors or []
        self.has_side_effect = has_side_effect

    def gen_shared_tensors(self, tree, **kwargs):
        self.shared_tensors = gpu_find_sharable_tensors(
            tree, self.stmt_table, self.tensor_table, **kwargs
        )

    @staticmethod
    def _produce_tensors(scope, tensors: Iterable[Tensor], body):
        for t in reversed(list(tensors)):
            body = t.build_tir_realize(scope, body)
        return body

    def parse(self, node, parent=None):
        if parent is None:
            if isinstance(node, ScheduleTree):
                self.gen_shared_tensors(node)
            body = super().parse(node, parent)
            write_tensors = set()
            for i in self.stmt_table.values():
                write_tensors = write_tensors.union(i.get_access()[1]['write'])
            return self._produce_tensors('', write_tensors, body)
        return super().parse(node, parent)

    def before_each_mark(self, node, ast_build):
        if node.name() == 'bind=threadIdx':
            for i in self.shared_tensors:
                i.gen_offset_ast(ast_build)
        pass

    def parse_mark(self, node, parent):
        mark = node.id().name()

        def _build_body():
            body = super(CUDANode2TIRParser, self).parse_mark(node, parent)
            if mark.startswith('bind=') and not isinstance(node.node(), isl.ast_node_for):
                _, axis = mark.rsplit('=', 1)
                with self.cuda_iter_var_table.axis(axis, 1) as iter_var:
                    body = tir.AttrStmt(node=iter_var, attr_key='thread_extent', value=tir_imm(1), body=body)
            return body

        if mark == 'bind=threadIdx' and mark not in self.mark_stack:
            def _under_shared(ith):
                if ith >= len(self.shared_tensors):
                    return _build_body()
                tensor = self.shared_tensors[ith]
                with self.tensor_table.scoped(tensor.origin.name, 'shared', tensor=tensor):
                    return tensor.build_tir_realize('shared', _under_shared(ith + 1))

            for i in self.shared_tensors:
                i.gen_offset_tvm_repr(self.expr_parser)
            body = _under_shared(0)
        else:
            body = _build_body()

        return body

    def parse_for(self, node, parent):
        if self.mark_stack and self.mark_stack[-1].startswith('bind='):
            _, axis = self.mark_stack[-1].rsplit('=', 1)
            bounds = []
            cur, cur_p = node, parent
            while isinstance(cur, isl.ast_node_for):
                bounds.append(self._for_bounds(cur))
                cur, cur_p = cur.body(), cur
            extents = [(upper - lower) // step for _, lower, upper, step in bounds]
            total = reduce(lambda x, y: x * y, extents)
            total = tir.ir_pass.Simplify(total)
            assert str(total).isdigit()
            total = int(str(total))
            anchors = [total // extents[0]]
            for i in range(1, len(bounds)):
                anchors.append(anchors[-1] // extents[i])

            def innermost():
                body = self.parse(cur, cur_p)
                if self.mark_stack[-1] == 'bind=threadIdx':
                    tensors_from_host = []
                    for i in self.shared_tensors:
                        if self.has_side_effect and 'read' in i.access_types \
                                or not self.has_side_effect and 'write' not in i.access_types:
                            tensors_from_host.append(i)
                    tensors_to_host = []
                    for i in self.shared_tensors:
                        if 'write' in i.access_types:
                            tensors_to_host.append(i)
                    stmts = []
                    for i in tensors_from_host:
                        stmts.append(i.build_copy_from_host(self.cuda_iter_var_table, self.iter_var_table))
                    if tensors_from_host:
                        stmts.append(tir.Evaluate(tir_sync('shared')))
                    stmts.append(body)
                    if tensors_to_host:
                        stmts.append(tir.Evaluate(tir_sync('shared')))
                    for i in tensors_to_host:
                        stmts.append(i.build_copy_to_host(self.cuda_iter_var_table, self.iter_var_table))
                    if len(stmts) >= 2:
                        body = tir.SeqStmt(stmts)
                return body

            def recur(num, axis_var):
                if num >= len(bounds):
                    return innermost()
                c_var_name, lower, upper, step = bounds[num]
                with self.iter_var_table.var(c_var_name) as c_var:
                    val = axis_var // anchors[num] % extents[num]
                    return tir.LetStmt(
                        c_var, val * step + lower,
                        body=recur(num + 1, axis_var)
                    )

            with self.cuda_iter_var_table.axis(axis, total) as iter_var:
                body = tir.AttrStmt(
                    node=iter_var, attr_key='thread_extent', value=tir_imm(total),
                    body=recur(0, iter_var)
                )

            return body
        return super().parse_for(node, parent)


@contextmanager
def building_poly(stmts, binds, arg_list):
    with threading.Lock():
        old_form_body = tvm.driver.build_module.form_body
        old_get_binds = tvm.driver.build_module.get_binds
        tvm.driver.build_module.form_body = lambda x: stmts
        tvm.driver.build_module.get_binds = lambda *_, **__: (binds, arg_list)
        yield
        tvm.driver.build_module.get_binds = old_get_binds
        tvm.driver.build_module.form_body = old_form_body


# noinspection PyUnresolvedReferences
def build_tvm_stmts(name, tree, parser: ISLNode2TIR, te_tensors=None):
    stmts = parser.parse(tree)
    stmts = tir.ir_pass.CanonicalSimplify(stmts)

    if te_tensors is None:
        tensor_table = parser.tensor_table
        te_tensors = [i.te_tensor for i in tensor_table.table.values()]

    binds, arg_list = tvm.driver.build_module.get_binds(te_tensors)

    with building_poly(stmts, binds, arg_list):
        tvm_s = te.create_schedule(te_tensors[-1].op)
        stmts = tvm.lower(tvm_s, te_tensors, name=slugify(name))

    # print(stmts)

    return stmts, te_tensors


'''
from ..polyhedral.statement import example_tensor_table, example_statements
from ..polyhedral.schedule_tree import ScheduleTree

parser = CUDANode2TIRParser(
    tensor_table=example_tensor_table,
    stmt_table=example_statements
)

tree = example_tree.copy()
tree.apply_params(n=N, m=M, q=Q)
tree.gpu_tile([1, 355])

print(tree.to_yaml())

stmts, _ = build_tvm_stmts('example_to_tvm', tree, parser)
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

evaluator = func.time_evaluator(func.entry_name, ctx, number=20)
print('poly speed:', evaluator(a, b, c).mean, end='\n\n')
'''
