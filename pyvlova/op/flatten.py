# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from tvm import topi

from .unary import UnaryElementwise
from ..poly import TensorTable


class Flatten2d(UnaryElementwise):
    calculated_args = {
        'out_channel': lambda channel=1, height=1, width=1, **_: channel * height * width,
    }

    @staticmethod
    def statements_factory(height=1, width=1, **_):
        def stmt(t, n, c, h, w):
            t['out'][n, c * height * width + h * width + w] = t['x'][n, c, h, w]

        return [stmt]

    @staticmethod
    def tensors_factory(batch=1, channel=1, height=1, width=1, **_):
        table = TensorTable()
        table.add_tensor('x', [batch, channel, height, width])
        table.add_tensor('out', [batch, channel * height * width])
        return table

    def topi_cuda_args(self, x=None, out=None):
        return [x, [self.batch, self.channel * self.height * self.width]]

    topi_cuda_calc_func = topi.reshape
    topi_cuda_schedule_func = topi.cuda.schedule_elemwise
