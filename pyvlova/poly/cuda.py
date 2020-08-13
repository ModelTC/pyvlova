from collections import defaultdict
from functools import reduce
from contextlib import contextmanager

import isl
from tvm import tir, te

from pyvlova.autotune.settings import cuda_settings
from pyvlova.poly.schedule_tree.node import SequenceNode, FilterNode, MarkNode, NodeWithSingleChild
from pyvlova.utils import tir_load, tir_imm, tir_store, structure_unnamed_fixed_box
from .poly import IterVarTable, Tensor, Statement, record_effective_op


class CUDAIterVarTable(IterVarTable):
    def __init__(self):
        super().__init__()
        self.axis_cnt = defaultdict(int)
        self.var_extents = defaultdict(lambda: defaultdict(lambda: 1))
        self.axis_extents = defaultdict(lambda: 1)
        self.axis_idx = defaultdict(lambda: tir_imm(0))

    @contextmanager
    def axis(self, name, extents):
        with self.var(name) as v:
            self.axis_extents[name] *= extents
            self.axis_idx[name] = self.axis_idx[name] * extents + v
            self.var_extents[name][str(v.var)] = extents
            yield v
            del self.var_extents[name][str(v.var)]
            self.axis_idx[name] = (self.axis_idx[name] - v) // extents
            self.axis_extents[name] //= extents

    def push(self, name=None, var=None):
        k = self.axis_cnt[name]
        self.axis_cnt[name] += 1
        name = f'{name}.{chr(ord("x") + k)}'
        if var is None:
            var = te.thread_axis(name)
        return super().push(name, var)

    def pop(self):
        var = super().pop()
        axis, _ = var.var.name.split('.', 1)
        self.axis_cnt[axis] -= 1
        return var


def cuda_tile(tree, tile_size, permutation=None):
    assert tree.parallel_tilable()
    box_size, lowers, strides = tree.outermost_band_box()
    n = len(box_size)
    tile_size = tile_size[:n]

    real_tile_size = [tile_size[i] * strides[i] for i in range(n)]
    filled_box_size = [-(-box_size[i] // (real_tile_size[i])) * real_tile_size[i] for i in range(n)]

    fake_args = ['i%d' % i for i in range(n)]

    thread_fake_constraints = [
        f'({i} mod {stride}) = (({lower}) mod {stride})'
        f' and 0 <= {i} - ({lower}) < {size}'
        for i, lower, stride, size in
        zip(fake_args, lowers, strides, filled_box_size)
    ]
    thread_fake_named_tuple = f'_thread[{", ".join(fake_args)}]'
    thread_fake_statement = isl.union_set(
        f'{{ {thread_fake_named_tuple} : {" and ".join(thread_fake_constraints)} }}'
    ).coalesce()

    block_fake_constraints = [
        f'({i} mod {stride}) = (({lower}) mod {stride})'
        f' and 0 <= {i} - ({lower}) < {size}'
        f' and ({i} mod {rt_size}) = (({lower}) mod {rt_size})'
        for i, lower, stride, size, rt_size in
        zip(fake_args, lowers, strides, filled_box_size, real_tile_size)
    ]
    block_fake_named_tuple = f'_block[{", ".join(fake_args)}]'
    block_fake_statement = isl.union_set(
        f'{{ {block_fake_named_tuple} : {" and ".join(block_fake_constraints)} }}'
    ).coalesce()

    old_domain = tree.domain()
    tree.add_to_domain(thread_fake_statement)
    tree.add_to_domain(block_fake_statement)

    band = tree.outermost_band()

    for i in range(n):
        s = band.schedule.at(i).union_add(
            isl.pw_aff(f'{{ {thread_fake_named_tuple} -> [({fake_args[i]})] }}'))
        band.schedule = band.schedule.set_at(i, s.coalesce())
        s = band.schedule.at(i).union_add(
            isl.pw_aff(f'{{ {block_fake_named_tuple} -> [({fake_args[i]})] }}'))
        band.schedule = band.schedule.set_at(i, s.coalesce())

    fake_branch = SequenceNode()
    fake_branch.add_child(FilterNode(filter='{%s}' % thread_fake_named_tuple))
    fake_branch.add_child(FilterNode(filter='{%s}' % block_fake_named_tuple))

    kernel_branch = FilterNode(filter=old_domain)
    if band.child:
        kernel_branch.child = band.child
    fake_branch.add_child(kernel_branch)

    band.child = fake_branch

    if permutation is not None:
        band.permute(*permutation)

    band.tile(*real_tile_size)
    band.insert_before(MarkNode('bind=blockIdx'))
    child = band.child
    child.insert_before(MarkNode('bind=threadIdx'))
    kernel = child.child
    kernel.insert_before(MarkNode('clear(bind)'))


class BlockTensorUsage(Tensor):
    def __init__(self, origin: Tensor, box_size, strides, offset_pma, access_types):
        self.origin = origin
        self.box_size = list(map(int, box_size))
        self.strides = list(map(int, strides))
        self.offset_pma: isl.pw_multi_aff = offset_pma.coalesce()
        assert isinstance(self.offset_pma, isl.pw_multi_aff)
        self.offset_ast = None
        self.offset_tvm_repr = None
        self.access_types = set(access_types)
        super().__init__(
            name=f'_{self.origin.name}_shared',
            shape=self.usage_extents(False),
            dtype=self.origin.dtype
        )

    def usage_extents(self, with_offset):
        extents = [-(-i // j) for i, j in zip(self.box_size, self.strides)]
        if not with_offset:
            return extents
        extents = [
            tir.Min(tir_imm(i), tir_imm(j) - k)
            for i, j, k in zip(extents, self.origin.shape, self.offset_tvm_repr)
        ]
        return extents
    
    def usage_extents_and_stride(self):
        extents = self.usage_extents(True)
        idx_strides = list(extents) + [1]
        for i in range(len(extents) - 1, -1, -1):
            idx_strides[i] *= idx_strides[i + 1]
        return extents, idx_strides

    def gen_offset_ast(self, ast_build: isl.ast_build):
        s_map = ast_build.schedule_map().flatten_domain()
        o_map = isl.map.from_pw_multi_aff(self.offset_pma).apply_domain(s_map)
        pma = isl.pw_multi_aff.from_union_map(o_map).as_pw_multi_aff()
        call = ast_build.call_from(pma)
        self.offset_ast = [call.arg(i) for i in range(1, call.n_arg())]

    def gen_offset_tvm_repr(self, expr_parser):
        assert self.offset_ast
        self.offset_tvm_repr = list(map(expr_parser.parse, self.offset_ast))
        self.offset = self.offset_tvm_repr
    
    def build_tir_realize(self, scope='shared', body=None):
        return super().build_tir_realize(scope=scope, body=body)

    def build_copy_from_host(self, cuda_var_table, iter_var_table):
        def copy_from_host(new_idx, old_idx):
            record_effective_op(tir_store(self.te_tensor, new_idx, self.origin[old_idx]))
        return self._build_copy(cuda_var_table, iter_var_table, copy_from_host)

    def build_copy_to_host(self, cuda_var_table, iter_var_table):
        def copy_to_host(new_idx, old_idx):
            self.origin[old_idx] = tir_load(self.te_tensor, new_idx)
        return self._build_copy(cuda_var_table, iter_var_table, copy_to_host)

    def _get_tensor_index(self, idx):
        extents, idx_strides = self.usage_extents_and_stride()
        if len(extents) >= 2:
            x = [
                idx // idx_strides[j + 1] % extents[j]
                for j in range(len(extents))
            ]
            y = [u + v for u, v in zip(x, self.offset_tvm_repr)]
        else:
            x, y = [idx], [idx + self.offset_tvm_repr[0]]
        return tuple(y), tuple(y)

    def _build_copy(self, cuda_var_table, iter_var_table, func):
        assert self.offset_tvm_repr

        def copy_shared(_, idx):
            func(*self._get_tensor_index(idx))

        stmt = Statement.from_calc(copy_shared)
        return self._build_copy_schedule(cuda_var_table, iter_var_table, stmt)

    def _build_copy_schedule(self, cuda_var_table: CUDAIterVarTable, iter_var_table: IterVarTable, stmt: Statement):
        num_threads = cuda_var_table.axis_extents['threadIdx']
        idx = cuda_var_table.axis_idx['threadIdx']
        total = reduce(tir.Mul, self.usage_extents(True))
        with iter_var_table.var() as iter_var, iter_var_table.var() as extent_var:
            body = tir.For(
                iter_var, tir_imm(0), extent_var, tir.For.Serial, 0,
                stmt.to_tvm(None, iter_var * num_threads + idx)
            )
            body = tir.LetStmt(extent_var, (total - 1 - idx) // num_threads + 1, body)
        return body


def check_cuda_tiled(tree):
    node = tree.root
    while node and isinstance(node, (NodeWithSingleChild, MarkNode)):
        if isinstance(node, MarkNode) and 'threadIdx' in node.mark:
            return True
        node = node.child
    return False


def cuda_find_sharable_tensors(tree, statements, tensors, max_shared_memory=None):
    node = tree.root

    while node and isinstance(node, (NodeWithSingleChild, MarkNode)):
        if isinstance(node, MarkNode) and 'threadIdx' in node.mark:
            break
        node = node.child
    assert isinstance(node, MarkNode) and 'threadIdx' in node.mark

    if max_shared_memory is None:
        max_shared_memory = cuda_settings['max_shared_memory']

    prefix = node.to_isl().prefix_schedule_relation()
    prefix = prefix.intersect_domain(tree.domain()).reverse()

    tensor_access = dict()
    tensor_stmts = defaultdict(lambda: defaultdict(lambda: isl.union_map('{}')))
    tensor_access_types = defaultdict(set)
    for _, stmt in statements.items():
        stmt_tensor_access, _ = stmt.get_access(tensors)
        for k in ('read', 'write'):
            for name, access in stmt_tensor_access[k].items():
                tensor_access_types[name].add(k)
                new_map = prefix.apply_range(access)
                assert new_map.isa_map()
                new_map = isl.map.from_union_map(new_map)
                if name in tensor_access:
                    new_map = new_map.add_map(tensor_access[name])
                    assert new_map.isa_map()
                    new_map = isl.map.from_union_map(new_map)
                tensor_access[name] = new_map
                tensor_stmts[k][name] = tensor_stmts[k][name].union(access.reverse())

    access_count = defaultdict(int)
    for name in tensor_access:
        for access_type in ('read', 'write'):
            ts_maps = tensor_stmts[access_type][name].intersect_range(tree.domain())
            stmt_maps = tensor_access[name].apply_range(ts_maps)
            stmts = list()
            stmt_maps.foreach_map(stmts.append)
            for stmt in stmts:
                box = stmt.range_simple_fixed_box_hull()
                box_size, _ = structure_unnamed_fixed_box(box)
                s = stmt.range()
                strides = [int(str(s.stride(i))) for i in range(len(box_size))]
                total = reduce(int.__mul__, [-(-i // j) for i, j in zip(box_size, strides)])
                access_count[name] += total

    usages = []
    for name in tensor_access:
        box = tensor_access[name].range_simple_fixed_box_hull()
        box_size, offset = structure_unnamed_fixed_box(box)
        s = tensor_access[name].range()
        strides = [int(str(s.stride(i))) for i in range(len(box_size))]
        usages.append(BlockTensorUsage(
            tensors[name], box_size, strides, offset, tensor_access_types[name]
        ))

    usages.sort(key=lambda x: x.size_in_bytes)

    res = []
    shared_total_usage = 0
    for i in usages:
        name = i.origin.name
        bytes_usage = i.size_in_bytes
        if bytes_usage * 8 >= access_count[name]:
            continue
        if bytes_usage + shared_total_usage > max_shared_memory:
            break
        shared_total_usage += bytes_usage
        res.append(i)

    return res
