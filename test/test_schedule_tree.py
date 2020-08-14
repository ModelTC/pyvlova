# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import unittest

from pyvlova.poly import cuda_tile

from .example import example_tree


class TestScheduleTree(unittest.TestCase):
    def test_cuda_tile(self):
        tree = example_tree.copy()
        tree.apply_params(n=1024, m=2048, q=4096)
        cuda_tile(tree, [391, 1096])


if __name__ == '__main__':
    unittest.main()
