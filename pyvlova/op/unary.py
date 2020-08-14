# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from tvm import te, topi

from .base import ArgumentedOp
from ..poly import TensorTable, Statement, trace_mode, ScheduleTree
from ..utils import tir_imm


def schedule(**kwargs):
    stmt = 'stmt[n, c, h, w]'
    constraints = '0 <= n < batch and 0 <= c < channel ' \
                  'and 0 <= h < height and 0 <= w < width'
    domain = '[batch, channel, height, width] -> {' \
             f'{stmt}: {constraints}' \
             '}'
    outer_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{stmt}->[({x})]}}', ('n', 'c', 'h', 'w')))

    tree = ScheduleTree.from_yaml(f'''
    domain: "{domain}"
    child:
        schedule: "{outer_schedule}"
        permutable: 1
        coincident: [1, 1, 1, 1]
    ''')
    tree.apply_params(**kwargs)
    return tree


def tensors(batch=1, channel=1, height=1, width=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, channel, height, width])
    table.add_tensor('out', [batch, channel, height, width])
    return table


class UnaryElementwise(ArgumentedOp):
    required_args = [
        'channel', 'height', 'width',
    ]
    optional_args = {
        'batch': 1,
    }
    calculated_args = {}
    tensor_order = ['x', 'out']
    inputs = ['x']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    topi_cuda_calc_ret_map = ['out']

    def topi_cuda_args(self, x=None, out=None):
        return [x]

    topi_cuda_schedule_func = topi.cuda.schedule_elemwise


class ReLU(UnaryElementwise):
    @staticmethod
    def statements_factory(**_):
        def stmt(t, n, c, h, w):
            if trace_mode.mode == 'tvm':
                t['out'][n, c, h, w] = te.max(t['x'][n, c, h, w], tir_imm(0.0))
            elif trace_mode.mode == 'tensor_access':
                t['out'][n, c, h, w] = t['x'][n, c, h, w]
            else:
                t['out'][n, c, h, w] = max(t['x'][n, c, h, w], 0.0)

        res = {}
        for i in [stmt]:
            res[i.__name__] = Statement.from_calc(i)
        return res

    topi_cuda_calc_func = topi.nn.relu


class ReLU6(UnaryElementwise):
    @staticmethod
    def statements_factory(**_):
        def stmt(t, n, c, h, w):
            if trace_mode.mode == 'tvm':
                t['out'][n, c, h, w] = te.max(te.min(t['x'][n, c, h, w], tir_imm(6.0)), tir_imm(0.0))
            elif trace_mode.mode == 'tensor_access':
                t['out'][n, c, h, w] = t['x'][n, c, h, w]
            else:
                t['out'][n, c, h, w] = max(min(t['x'][n, c, h, w], 6.0), 0.0)

        res = {}
        for i in [stmt]:
            res[i.__name__] = Statement.from_calc(i)
        return res

    topi_cuda_calc_func = lambda x: topi.maximum(topi.minimum(x, tir_imm(6.0)), tir_imm(0.0))
