"""Parse ISL AST to TIR"""

from contextlib import contextmanager
from functools import reduce
from typing import Dict, Iterable, List
import re

import tvm
from tvm import tir
import isl

from ..poly import cuda_find_sharable_tensors, BlockTensorUsage, IterVarTable, CUDAIterVarTable, TensorTable, Statement, Tensor, ScheduleTree, check_cuda_tiled
from ..utils import tir_imm, slugify, tir_cuda_shared_sync, tir_thread_extent_attr


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

    def parse_op_pdiv_r(self, expr, parent):
        return tir.Mod(self.parse(expr.arg(0), expr), self.parse(expr.arg(1), expr))

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
        if not then_node and not else_node:
            return None
        return tir.IfThenElse(condition=cond, then_case=then_node, else_case=else_node)


class CUDAISLNode2TIR(ISLNode2TIR):
    def __init__(self, cuda_iter_var_table=None, shared_tensors=None,
                 has_side_effect=False, do_shared_opt=True, **kwargs):
        super().__init__(**kwargs)
        self.cuda_iter_var_table: CUDAIterVarTable = cuda_iter_var_table or CUDAIterVarTable()
        self.set_shared_tensors(shared_tensors)
        self.has_side_effect = has_side_effect
        self.do_shared_opt = do_shared_opt
    
    def set_shared_tensors(self, shared_tensors):
        if shared_tensors:
            self.shared_tensors: List[BlockTensorUsage] = list(shared_tensors)
            self._shared_tensors_set_flag = True
        else:
            self.shared_tensors: List[BlockTensorUsage] = []
            self._shared_tensors_set_flag = False

    def _gen_shared_tensors(self, tree, **kwargs):
        if not self._shared_tensors_set_flag:
            self.shared_tensors = cuda_find_sharable_tensors( tree, self.stmt_table, self.tensor_table, **kwargs)

    def _reset_shared_tensors(self):
        if not self._shared_tensors_set_flag:
            self.shared_tensors = []

    def parse(self, node, parent=None):
        if parent is None:
            if isinstance(node, ScheduleTree):
                assert check_cuda_tiled(node), 'schedule tree need to be tiled first'
                if self.do_shared_opt:
                    self._gen_shared_tensors(node)
                else:
                    self._reset_shared_tensors()
            stmt = super().parse(node, parent)
            return stmt
        return super().parse(node, parent)

    def before_each_mark(self, node, ast_build):
        if 'threadIdx' in node.name():
            for tensor_usage in self.shared_tensors:
                tensor_usage.gen_offset_ast(ast_build)

    def parse_mark(self, node, parent):
        mark = node.id().name()

        b = super().parse_mark(node, parent)
        if mark.startswith('bind=') and not isinstance(node.node(), isl.ast_node_for):
            _, axis = mark.rsplit('=', 1)
            if not re.fullmatch(r'\w+', axis):
                axis, *axis_per_bind = re.findall(r'\w+', axis)
                for i in reversed(axis_per_bind):
                    with self.cuda_iter_var_table.axis(i, 1) as iter_var:
                        b = tir_thread_extent_attr(iter_var, extent=1, body=b)
            with self.cuda_iter_var_table.axis(axis, 1) as iter_var:
                b = tir_thread_extent_attr(iter_var, extent=1, body=b)
        return b
    
    def parse_cuda_axis(self, node, parent):
        _, axis = self.mark_stack[-1].rsplit('=', 1)

        one_per_bind = False
        if not re.fullmatch(r'\w+', axis):
            one_per_bind = True
            axis, *axis_per_bind = re.findall(r'\w+', axis)

        bounds = []
        cur, cur_p = node, parent
        while isinstance(cur, isl.ast_node_for):
            bounds.append(self._for_bounds(cur))
            cur, cur_p = cur.body(), cur

        extents = [(upper - lower) // step for _, lower, upper, step in bounds]

        total = reduce(lambda x, y: x * y, extents)
        total = tvm.arith.analyzer.Analyzer().simplify(total)
        assert str(total).isdigit()
        total = int(str(total))

        anchors = [total // extents[0]]
        for idx in range(1, len(bounds)):
            anchors.append(anchors[-1] // extents[idx])

        def innermost():
            body = self.parse(cur, cur_p)
            if 'threadIdx' in self.mark_stack[-1]:
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
                    stmts.append(tir.Evaluate(tir_cuda_shared_sync()))
                stmts.append(body)
                if tensors_to_host:
                    stmts.append(tir.Evaluate(tir_cuda_shared_sync()))
                for i in tensors_to_host:
                    stmts.append(i.build_copy_to_host(self.cuda_iter_var_table, self.iter_var_table))
                if len(stmts) >= 2:
                    body = tir.SeqStmt(stmts)
            return body

        def _under_shared(ith):
            if ith >= len(self.shared_tensors):
                return innermost()
            tensor = self.shared_tensors[ith]
            with self.tensor_table.scoped(tensor.origin.name, 'shared', tensor=tensor):
                return tensor.build_tir_realize('shared', _under_shared(ith + 1))

        def expand_axis_var(num, axis_var):
            if num >= len(bounds):
                if 'threadIdx' in self.mark_stack[-1]:
                    for tensor_usage in self.shared_tensors:
                        tensor_usage.gen_offset_tvm_repr(self.expr_parser)
                    return _under_shared(0)
                else:
                    return innermost()
            c_var_name, lower, _, step = bounds[num]
            with self.iter_var_table.var(c_var_name) as c_var:
                val = axis_var // anchors[num] % extents[num]
                return tir.LetStmt(
                    c_var, val * step + lower,
                    body=expand_axis_var(num + 1, axis_var)
                )

        def wrap_one_per_bind(body):
            if one_per_bind:
                for i in reversed(axis_per_bind):
                    with self.cuda_iter_var_table.axis(i, 1) as iter_var:
                        body = tir_thread_extent_attr(iter_var, extent=1, body=body)
            return body

        with self.cuda_iter_var_table.axis(axis, total) as iter_var:
            body = wrap_one_per_bind(expand_axis_var(0, iter_var))
            return tir_thread_extent_attr(iter_var, extent=total, body=body)

    def parse_for(self, node, parent):
        if self.mark_stack and self.mark_stack[-1].startswith('bind='):
            return self.parse_cuda_axis(node, parent)
        return super().parse_for(node, parent)
