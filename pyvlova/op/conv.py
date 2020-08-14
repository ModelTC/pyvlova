# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
from tvm import topi

from .base import ArgumentedOp, OpParameter, CombinedOp
from .binary import ChannelwiseAdd
from .padding import Padding
from ..poly import TensorTable, Statement, ScheduleTree


def schedule(**kwargs):
    init_t = 'stmt_init[n, c, h, w]'
    calc_t = 'stmt_calc[n, c, h, w, i, j, k]'
    output_constraints = '0 <= n < batch and 0 <= c < out_channel ' \
                         'and 0 <= h < out_height and 0 <= w < out_width'
    calc_constraints = '0 <= i < in_channel and 0 <= j < kernel_height and 0 <= k < kernel_width'
    domain = '[batch, in_channel, in_height, in_width, out_channel, out_height, out_width, ' \
             'kernel_height, kernel_width] -> {' \
             f'{init_t}: {output_constraints}; ' \
             f'{calc_t}: {output_constraints} and {calc_constraints}' \
             '}'
    outer_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{init_t}->[({x})];{calc_t}->[({x})]}}', ('n', 'c', 'h', 'w')))
    inner_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{calc_t}->[({x})]}}', ('i', 'j', 'k')))

    tree = ScheduleTree.from_yaml(f'''
    domain: "{domain}"
    child:
        schedule: "{outer_schedule}"
        permutable: 1
        coincident: [1, 1, 1, 1]
        child:
            sequence:
              - filter: "{{{init_t}}}"
              - filter: "{{{calc_t}}}"
                child:
                    schedule: "{inner_schedule}"
                    permutable: 1
                    coincident: [1, 1, 1]
    ''')
    tree.apply_params(**kwargs)
    return tree


def tensors(batch=1, in_channel=1, in_height=1, in_width=1, out_channel=1,
            out_height=1, out_width=1, kernel_height=1, kernel_width=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, in_channel, in_height, in_width])
    table.add_tensor('weight', [out_channel, in_channel, kernel_height, kernel_width])
    table.add_tensor('out', [batch, out_channel, out_height, out_width])
    return table


def statements(stride_height=1, stride_width=1, **_):
    def stmt_init(t, n, c, h, w):
        t['out'][n, c, h, w] = 0.0

    def stmt_calc(t, n, c, h, w, i, j, k):
        t['out'][n, c, h, w] = t['out'][n, c, h, w] \
            + t['x'][n, i, h * stride_height + j, w * stride_width + k] * t['weight'][c, i, j, k]

    res = {}
    for f in [stmt_init, stmt_calc]:
        res[f.__name__] = Statement.from_calc(f)
    return res


class PlainConv2d(ArgumentedOp):
    required_args = [
        'in_channel', 'in_height', 'in_width', 'out_channel',
        'kernel_height', 'kernel_width',
    ]
    optional_args = {
        'batch': 1, 'stride_height': 1, 'stride_width': 1
    }
    calculated_args = {
        'out_height': lambda **a: (a['in_height'] - a['kernel_height']) // a['stride_height'] + 1,
        'out_width': lambda **a: (a['in_width'] - a['kernel_width']) // a['stride_width'] + 1,
    }
    tensor_order = ['x', 'weight', 'out']
    inputs = ['x', 'weight']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    statements_factory = statements
    topi_cuda_task_name = 'conv2d_nchw.cuda'

    def topi_cuda_args(self, x=None, weight=None, out=None):
        return [x, weight, [self.stride_height, self.stride_width], 0, 1, out.dtype]

    topi_cuda_calc_func = topi.cuda.conv2d_nchw
    topi_cuda_schedule_func = topi.cuda.schedule_conv2d_nchw
    topi_cuda_calc_ret_map = ['out']


class Conv2d(CombinedOp):
    weight = OpParameter('weight')
    bias = OpParameter('bias')

    def __init__(self, batch=1, in_channel=1, in_height=1, in_width=1,
                 out_channel=1, kernel_height=1, kernel_width=1, stride_height=1, stride_width=1,
                 pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
                 biased=False, name=''):
        super().__init__(name=name)
        if pad_top or pad_bottom or pad_left or pad_right:
            self.pad = Padding(
                name=self.name + '.pad', batch=batch,
                channel=in_channel, in_height=in_height, in_width=in_width,
                pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right
            )
            self._ops.append(self.pad)
            in_height = self.pad.out_height
            in_width = self.pad.out_width
        else:
            self.pad = None
        self.conv = PlainConv2d(
            name=self.name + '.conv', batch=batch,
            in_channel=in_channel, in_height=in_height, in_width=in_width,
            out_channel=out_channel, kernel_height=kernel_height, kernel_width=kernel_width,
            stride_height=stride_height, stride_width=stride_width
        )
        self._ops.append(self.conv)
        self.weight = self.conv.tensors['weight']
        if biased:
            self.bias_layer = ChannelwiseAdd(
                name=self.name + '.bias_layer',
                batch=self.conv.batch, channel=self.conv.out_channel,
                height=self.conv.out_height, width=self.conv.out_width
            )
            self._ops.append(self.bias_layer)
            self.bias = self.bias_layer.tensors['y']
        else:
            self.bias_layer = None

        for i in ['batch', 'out_channel', 'out_height', 'out_width']:
            setattr(self, i, getattr(self.conv, i))

    def calc(self, x):
        if self.pad is not None:
            x = self.pad.calc(x)
        x = self.conv.calc(x, self.weight)
        if self.bias_layer is not None:
            x = self.bias_layer.calc(x, self.bias)
        return x
