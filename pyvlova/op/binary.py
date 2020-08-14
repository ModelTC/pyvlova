# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from tvm import topi

from .base import ArgumentedOp
from .unary import schedule
from ..poly import TensorTable


def tensors(batch=1, channel=1, height=1, width=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, channel, height, width])
    table.add_tensor('y', [batch, channel, height, width])
    table.add_tensor('out', [batch, channel, height, width])
    return table


class BinaryElementwise(ArgumentedOp):
    required_args = [
        'channel', 'height', 'width',
    ]
    optional_args = {
        'batch': 1,
    }
    calculated_args = {}
    tensor_order = ['x', 'y', 'out']
    inputs = ['x', 'y']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    topi_cuda_calc_ret_map = ['out']

    def topi_cuda_args(self, x=None, y=None, out=None):
        return [x, y]

    topi_cuda_schedule_func = topi.cuda.schedule_elemwise


class ElementwiseAdd(BinaryElementwise):
    @staticmethod
    def statements_factory(**_):
        def stmt(t, n, c, h, w):
            t['out'][n, c, h, w] = t['x'][n, c, h, w] + t['y'][n, c, h, w]

        return [stmt]

    topi_cuda_calc_func = topi.add


class BinaryChannelwise(BinaryElementwise):
    @staticmethod
    def tensors_factory(batch=1, channel=1, height=1, width=1, **_):
        table = TensorTable()
        table.add_tensor('x', [batch, channel, height, width])
        table.add_tensor('y', [channel])
        table.add_tensor('out', [batch, channel, height, width])
        return table


class ChannelwiseAdd(BinaryChannelwise):
    @staticmethod
    def statements_factory(**_):
        def stmt(t, n, c, h, w):
            t['out'][n, c, h, w] = t['x'][n, c, h, w] + t['y'][c]

        return [stmt]

    def topi_cuda_args(self, x=None, y=None, out=None):
        return [self.channel, x, y]

    @staticmethod
    def topi_cuda_calc_func(channel, x, y):
        return topi.add(x, topi.reshape(y, (channel, 1, 1)))
