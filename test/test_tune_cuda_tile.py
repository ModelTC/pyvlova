import unittest

from pyvlova.autotune.cuda import tune_cuda_tile
from pyvlova.codegen import CUDAISLNode2TIR

from .example import example_tree, get_example_tensor_table, example_stmt_table


class TestTuneCUDATile(unittest.TestCase):
    def test_cuda_tile(self):
        tree = example_tree.copy()
        n, m, q = 13, 17, 19
        tree.apply_params(n=n, m=m, q=q)
        tensor_table = get_example_tensor_table(n, m, q)
        parser = CUDAISLNode2TIR(stmt_table=example_stmt_table, tensor_table=tensor_table)
        args = [tensor_table['A'].te_tensor, tensor_table['B'].te_tensor, tensor_table['C'].te_tensor]
        tune_cuda_tile('test', tree, args, parser)


if __name__ == '__main__':
    unittest.main()
