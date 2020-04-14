from collections import defaultdict
from functools import reduce

import isl
import tvm
from tvm import tir

from pyvlova.autotune.settings import cuda_settings
from pyvlova.poly.poly import Tensor, CUDAIterVarTable, IterVarTable, Statement, record_effective_op
from pyvlova.poly.schedule_tree.node import DomainNode, ExtensionNode, SequenceNode, FilterNode, MarkNode, \
    NodeWithSingleChild
from pyvlova.utils import get_unnamed_tuples, tir_load, tir_imm, tir_store, structure_unnamed_fixed_box, \
    map_out_constant_dim, tir_sync, copy_ast_build


def gpu_tile(tree, tile_size, permutation=None):
    assert tree.parallel_tilable()
    box_size, lowers, strides = tree.outermost_band_box()
    n = len(box_size)
    assert len(tile_size) == n

    real_tile_size = [tile_size[i] * strides[i] for i in range(n)]
    filled_box_size = [-(-box_size[i] // (real_tile_size[i])) * real_tile_size[i] for i in range(n)]

    thread_fake_args = ['i%d' % i for i in range(n)]
    thread_fake_constraints = [
        f'({i} mod {stride}) = (({lower}) mod {stride})'
        f' and 0 <= {i} - ({lower}) < {size}'
        for i, lower, stride, size in
        zip(thread_fake_args, lowers, strides, filled_box_size)
    ]
    thread_fake_named_tuple = f'_thread[{", ".join(thread_fake_args)}]'
    thread_fake_statement = isl.union_set(
        f'{{ {thread_fake_named_tuple} : {" and ".join(thread_fake_constraints)} }}'
    ).coalesce()

    old_domain = tree.domain()
    if isinstance(tree.root, DomainNode):
        tree.root.domain = tree.root.domain.union(thread_fake_statement)
    elif isinstance(tree.root, ExtensionNode):
        tree.root.extension = tree.root.extension.union(
            isl.union_map.from_range(thread_fake_statement))
    else:
        assert False

    band = tree.outermost_band()

    for i in range(n):
        s = band.schedule.at(i).union_add(
            isl.pw_aff(f'{{ {thread_fake_named_tuple} -> [({thread_fake_args[i]})] }}'))
        band.schedule = band.schedule.set_at(i, s.coalesce())

    thread_fake_branch = SequenceNode()
    thread_fake_branch.add_child(FilterNode(filter='{%s}' % thread_fake_named_tuple))
    old_branch = thread_fake_branch.add_child(FilterNode(filter=old_domain))
    if band.child:
        old_branch.child = band.child
    band.child = thread_fake_branch

    if permutation is not None:
        band.permute(*permutation)

    band.tile(*real_tile_size)

    band.insert_before(MarkNode('bind=blockIdx'))
    child = band.child
    child.insert_before(MarkNode('bind=threadIdx'))
    kernel = child.child
    kernel.insert_before(MarkNode('clear(bind)'))


class BlockTensorUsage(Tensor):
    def __init__(self, origin: Tensor, box_size, strides, offset_map, access_types):
        self.origin = origin
        self.box_size = list(box_size)
        self.strides = list(strides)
        self.offset_map: isl.map = offset_map.coalesce()
        assert isinstance(self.offset_map, isl.map)
        self.offset_ast = None
        self.offset_tvm_repr = None
        self.access_types = set(access_types)
        super(BlockTensorUsage, self).__init__(
            name=f'_{self.origin.name}_shared',
            shape=self.extent,
            dtype=self.origin.dtype
        )

    @property
    def extent(self):
        return [-(-i // j) for i, j in zip(self.box_size, self.strides)]

    def build_tir_realize(self, scope='shared', body=None):
        return super(BlockTensorUsage, self).build_tir_realize(scope, body)

    def gen_offset_ast(self, ast_build: isl.ast_build):
        ast_build = copy_ast_build(ast_build)
        s_map = isl.set(str(ast_build.schedule_space())).identity().flatten_range()
        upma = isl.union_pw_multi_aff.from_union_map(s_map.apply_range(self.offset_map))
        assert upma.isa_pw_multi_aff()
        pma = upma.as_pw_multi_aff()
        call = ast_build.call_from(pma)
        self.offset_ast = [call.arg(i) for i in range(1, call.n_arg())]

    def gen_offset_tvm_repr(self, expr_parser):
        assert self.offset_ast
        self.offset_tvm_repr = list(map(expr_parser.parse, self.offset_ast))

    UNROLL_SIZE = 8

    def build_copy_from_host(self, cuda_var_table, iter_var_table):
        def copy_from_host(new_idx, old_idx):
            record_effective_op(tir_store(self.te_tensor, new_idx, self.origin[old_idx]))
        return self._build_copy(cuda_var_table, iter_var_table, copy_from_host)

    def build_copy_to_host(self, cuda_var_table, iter_var_table):
        def copy_to_host(new_idx, old_idx):
            self.origin[old_idx] = tir_load(self.te_tensor, new_idx)
        return self._build_copy(cuda_var_table, iter_var_table, copy_to_host)

    def _get_tensor_index(self, idx):
        extent = self.extent
        idx_strides = list(extent) + [1]
        for i in range(len(extent) - 1, -1, -1):
            idx_strides[i] *= idx_strides[i + 1]
        if len(extent) >= 2:
            x = [
                idx // idx_strides[j + 1] % extent[j]
                for j in range(len(extent))
            ]
            x[0] = idx // idx_strides[1]
            x[-1] = idx % extent[-1]
            y = [u + v for u, v in zip(x, self.offset_tvm_repr)]
            return tuple(x), tuple(y)
        else:
            return (idx,), (idx + self.offset_tvm_repr[0],)

    def _build_copy(self, cuda_var_table, iter_var_table, func):
        assert self.offset_tvm_repr

        def copy_shared(_, idx):
            func(*self._get_tensor_index(idx))

        stmt = Statement.from_calc(copy_shared)
        return self._build_copy_schedule(cuda_var_table, iter_var_table, stmt)

    def _build_copy_schedule(self, cuda_var_table: CUDAIterVarTable, iter_var_table: IterVarTable, stmt: Statement):
        num_threads = cuda_var_table.axis_extent['threadIdx']
        idx = cuda_var_table.axis_idx['threadIdx']
        total = reduce(int.__mul__, self.extent)
        with iter_var_table.var() as iter_var:
            _, old_idx = self._get_tensor_index(iter_var * num_threads + idx)
            # noinspection PyTypeChecker
            cond = reduce(tir.And, [
                tir.LT(i - j, tir_imm(k))
                for i, j, k in zip(old_idx, self.origin.offset, self.origin.shape)
            ])
            # noinspection PyTypeChecker
            body = tir.For(
                iter_var, tir_imm(0), -(-total // num_threads), tir.For.Serial, 0,
                tir.IfThenElse(
                    cond, stmt.to_tvm(None, iter_var * num_threads + idx), None
                )
            )
        return body

    def _bad_build_copy_schedule(self, cuda_var_table: CUDAIterVarTable, iter_var_table: IterVarTable, stmt: Statement):
        raise Exception('Bad implementation')
        # noinspection PyUnreachableCode
        num_threads = cuda_var_table.axis_extent['threadIdx']
        idx = cuda_var_table.axis_idx['threadIdx']
        total = reduce(int.__mul__, self.extent)

        def _build_slow_loop(extent, fast_body):
            last_idx = (extent - 1) * num_threads + idx
            _, last_old_idx = self._get_tensor_index(last_idx)
            # noinspection PyTypeChecker
            slow_cond = reduce(tir.Or, [
                tir.GE(i - j, tir_imm(k))
                for i, j, k in zip(last_old_idx, self.origin.offset, self.origin.shape)
            ])
            with iter_var_table.var() as iter_var:
                _, old_idx = self._get_tensor_index(iter_var * num_threads + idx)
                # noinspection PyTypeChecker
                cond = reduce(tir.And, [
                    tir.LT(i - j, tir_imm(k))
                    for i, j, k in zip(old_idx, self.origin.offset, self.origin.shape)
                ])
                # noinspection PyTypeChecker
                slow_body = tir.For(
                    iter_var, tir_imm(0), extent, tir.For.Serial, 0,
                    tir.IfThenElse(
                        cond, stmt.to_tvm(None, iter_var * num_threads + idx), None
                    )
                )
            return tir.IfThenElse(slow_cond, slow_body, fast_body)

        def _build_opt_loop(extent):
            if not extent:
                return None
            outer_extent = extent // self.UNROLL_SIZE
            remained_extent = extent % self.UNROLL_SIZE
            with iter_var_table.var() as iter_var:
                remained_body = tir.For(
                    iter_var, tir_imm(outer_extent * self.UNROLL_SIZE),
                    remained_extent, tir.For.Unrolled, 0,
                    stmt.to_tvm(None, iter_var * num_threads + idx)
                )
            if outer_extent:
                with iter_var_table.var() as outer_var:
                    with iter_var_table.var() as inner_var:
                        loop_body = stmt.to_tvm(
                            None, (outer_var * self.UNROLL_SIZE + inner_var) * num_threads + idx
                        )
                        loop_body = tir.For(
                            inner_var, tir_imm(0), self.UNROLL_SIZE, tir.For.Unrolled, 0, loop_body
                        )
                        loop_body = tir.For(
                            outer_var, tir_imm(0), outer_extent, tir.For.Serial, 0, loop_body
                        )
                return tir.SeqStmt([loop_body, remained_body])
            return remained_body

        def _build_loop(extent):
            return _build_slow_loop(extent, _build_opt_loop(extent))

        if total % num_threads:
            return tir.IfThenElse(
                tir.LT(idx, tir_imm(total % num_threads)),
                _build_loop(-(-total // num_threads)),
                _build_loop(total // num_threads)
            )
        return _build_loop(total // num_threads)

    def getitem_tvm(self, key):
        key = tuple((i - j for i, j in zip(key, self.offset_tvm_repr)))
        return super(BlockTensorUsage, self).getitem_tvm(key)

    def setitem_tvm(self, key, value):
        key = tuple((i - j for i, j in zip(key, self.offset_tvm_repr)))
        super(BlockTensorUsage, self).setitem_tvm(key, value)


def gpu_find_sharable_tensors(tree, statements, tensors, max_shared_memory=None):
    node = tree.root

    while node and isinstance(node, (NodeWithSingleChild, MarkNode)):
        if isinstance(node, MarkNode) and node.mark == 'bind=threadIdx':
            break
        node = node.child
    if not isinstance(node, MarkNode) or node.mark != 'bind=threadIdx':
        return None

    if max_shared_memory is None:
        max_shared_memory = cuda_settings['max_shared_memory']

    prefix = node.to_isl().prefix_schedule_relation()
    prefix = prefix.intersect_domain(tree.domain()).reverse()

    tensor_access = dict()
    tensor_stmts = defaultdict(lambda: defaultdict(lambda: isl.union_map('{}')))
    tensor_access_types = defaultdict(set)
    for stmt_name, stmt in statements.items():
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

    schedule_space = tree.domain().apply(node.to_isl().prefix_schedule_relation()).coalesce()
    real_space_map = map_out_constant_dim(schedule_space)

    usages = []
    for name in tensor_access:
        box = tensor_access[name].range_simple_fixed_box_hull()
        box_size, offset = structure_unnamed_fixed_box(box)
        offset_map = isl.map.from_pw_multi_aff(offset)
        offset_map = real_space_map.reverse().apply_range(offset_map)
        s = tensor_access[name].range()
        strides = [int(str(s.stride(i))) for i in range(len(box_size))]
        usages.append(BlockTensorUsage(
            tensors[name], box_size, strides, offset_map, tensor_access_types[name]
        ))

    usages.sort(key=lambda x: x.size_in_bytes)

    res = []
    shared_total_usage = 0
    for i in usages:
        name = i.origin.name
        bytes_usage = i.size_in_bytes
        if bytes_usage * 3 >= access_count[name]:
            continue
        if bytes_usage + shared_total_usage > max_shared_memory:
            break
        shared_total_usage += bytes_usage
        res.append(i)

    res = list(filter(lambda x: x.access_types == {'read'}, res))

    return res
