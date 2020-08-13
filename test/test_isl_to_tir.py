import unittest

import tvm
import numpy as np

from pyvlova.poly import ScheduleTree, cuda_tile
from pyvlova.codegen import ISLNode2TIR, CUDAISLNode2TIR, build_tvm_stmt

from .example import example_tree, example_stmt_table, get_example_tensor_table


class TestISL2TIR(unittest.TestCase):
    def test_cpu_to_tir(self):
        tree = example_tree.copy()
        n, m, q = 121, 391, 214
        tree.apply_params(n=n, m=m, q=q)
        tensor_table = get_example_tensor_table(n, m, q)
        parser = ISLNode2TIR(stmt_table=example_stmt_table, tensor_table=tensor_table)
        stmt = parser.parse(tree)
        stmt_repr_expect = '''
let _isl_c0_extent = floordiv(((120 + 1) - 0), 1)
for (_i0, 0, _isl_c0_extent) {
  let c0 = ((_i0*1) + 0)
  let _isl_c1_extent = floordiv(((390 + 1) - 0), 1)
  for (_i3, 0, _isl_c1_extent) {
    let c1 = ((_i3*1) + 0)
    C[c0, c1] =0f
    let _isl_c2_extent = floordiv(((213 + 1) - 0), 1)
    for (_i6, 0, _isl_c2_extent) {
      let c2 = ((_i6*1) + 0)
      C[c0, c1] =(C[c0, c1] + (A[c0, c2]*B[c2, c1]))
    }
  }
}'''
        self.assertEqual(str(stmt).strip(), stmt_repr_expect.strip())
        self._check_matmul_kernel(stmt, tensor_table, tvm.cpu())

    def test_cuda_to_tir(self):
        tree = example_tree.copy()
        n, m, q = 121, 391, 214
        tree.apply_params(n=n, m=m, q=q)
        cuda_tile(tree, [21, 38])
        tensor_table = get_example_tensor_table(n, m, q)
        parser = CUDAISLNode2TIR(stmt_table=example_stmt_table, tensor_table=tensor_table)
        stmt = parser.parse(tree)
        self._check_matmul_kernel(stmt, tensor_table, tvm.gpu(), target='cuda')

    def test_cuda_to_tir_edge_case_1(self):
        tree = example_tree.copy()
        n, m, q = 13, 17, 19
        tree.apply_params(n=n, m=m, q=q)
        cuda_tile(tree, [13, 17])
        tensor_table = get_example_tensor_table(n, m, q)
        parser = CUDAISLNode2TIR(stmt_table=example_stmt_table, tensor_table=tensor_table)
        stmt = parser.parse(tree)
        self._check_matmul_kernel(stmt, tensor_table, tvm.gpu(), target='cuda')

    def test_cuda_to_tir_edge_case_2(self):
        tree = example_tree.copy()
        n, m, q = 13, 17, 19
        tree.apply_params(n=n, m=m, q=q)
        cuda_tile(tree, [1, 1])
        tensor_table = get_example_tensor_table(n, m, q)
        parser = CUDAISLNode2TIR(stmt_table=example_stmt_table, tensor_table=tensor_table)
        stmt = parser.parse(tree)
        self._check_matmul_kernel(stmt, tensor_table, tvm.gpu(), target='cuda')

    def _check_matmul_kernel(self, stmt, tensor_table, ctx, target=None):
        kernel = build_tvm_stmt(stmt, [tensor_table['A'].te_tensor, tensor_table['B'].te_tensor, tensor_table['C'].te_tensor], target=target)
        a_np = np.random.uniform(size=tensor_table['A'].shape).astype(tensor_table['A'].dtype)
        b_np = np.random.uniform(size=tensor_table['B'].shape).astype(tensor_table['B'].dtype)
        c_np = a_np @ b_np
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.empty(c_np.shape, ctx=ctx)
        kernel(a, b, c)
        tvm.testing.assert_allclose(c_np, c.asnumpy(), rtol=1e-5)
        evaluator = kernel.time_evaluator(kernel.entry_name, ctx, number=20)
        evaluator(a, b, c)


if __name__ == '__main__':
    unittest.main()
