# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import unittest

import tvm
import numpy

from pyvlova.op import *


n_trials = 8


class TestOp(unittest.TestCase):
    def setUp(self):
        self.ctx = tvm.gpu()
    
    def _get_array(self, shape):
        return tvm.nd.array(numpy.random.random(shape).astype('float32'), ctx=self.ctx)
    
    def _cuda_test_op(self, op, rtol=1e-7, atol=1e-7):
        args = []
        for name in op.inputs:
            args.append(self._get_array(op.tensors[name].shape))
        with calc_mode.under('tvm_cuda_timing'):
            op.imp(tune_kwargs={'n_trial': n_trials})
            out_a = op.calc(*args)
        with calc_mode.under('tvm_topi_cuda_timing'):
            op.imp(tune_kwargs={'n_trial': n_trials})
            out_b = op.calc(*args)
        if not isinstance(out_a, tuple):
            out_a, out_b = (out_a,), (out_b,)
        for u, v in zip(out_a, out_b):
            tvm.testing.assert_allclose(u.asnumpy(), u.asnumpy(), rtol, atol)

    def test_relu(self):
        relu = ReLU(channel=64, height=224, width=224)
        self._cuda_test_op(relu)

    def test_adaptive_pool(self):
        adpavgpool = AdaptivePool(
            batch=7, channel=64, in_height=224, in_width=224,
            out_height=1, out_width=1, pool_type='avg'
        )
        self._cuda_test_op(adpavgpool)

    def test_conv2d(self):
        conv = PlainConv2d(
            batch=5,
            in_channel=64, in_height=24, in_width=24,
            out_channel=32, kernel_height=7, kernel_width=7,
            stride_height=2, stride_width=2
        )
        self._cuda_test_op(conv)

    def test_linear(self):
        linear = PlainLinear(in_channel=64, out_channel=8)
        self._cuda_test_op(linear)

    def test_grouped_conv2d(self):
        conv = PlainGroupedConv2d(
            batch=7,
            in_channel=64, in_height=24, in_width=24,
            out_channel=32, kernel_height=7, kernel_width=7,
            stride_height=2, stride_width=2, groups=32
        )
        self._cuda_test_op(conv)


if __name__ == '__main__':
    unittest.main()
