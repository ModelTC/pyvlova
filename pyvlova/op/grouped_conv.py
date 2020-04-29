import topi

from pyvlova.op.conv import PlainConv2d, Conv2d
from pyvlova.poly.schedule_tree.tree import ScheduleTree
from ..poly.poly import TensorTable, Statement


def schedule(**kwargs):
    init_t = 'stmt_init[n, c, h, w]'
    calc_t = 'stmt_calc[n, c, h, w, i, j, k]'
    output_constraints = '0 <= n < batch and 0 <= c < out_channel ' \
                         'and 0 <= h < out_height and 0 <= w < out_width'
    calc_constraints = '0 <= i < in_group_size and 0 <= j < kernel_height and 0 <= k < kernel_width'
    domain = '[batch, in_channel, in_height, in_width, out_channel, out_height, out_width, ' \
             'kernel_height, kernel_width, in_group_size] -> {' \
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
            out_height=1, out_width=1, kernel_height=1, kernel_width=1, in_group_size=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, in_channel, in_height, in_width])
    table.add_tensor('weight', [out_channel, in_group_size, kernel_height, kernel_width])
    table.add_tensor('out', [batch, out_channel, out_height, out_width])
    return table


def statements(stride_height=1, stride_width=1, in_group_size=1, out_group_size=1, **_):
    def stmt_init(t, n, c, h, w):
        t['out'][n, c, h, w] = 0.0

    def stmt_calc(t, n, c, h, w, i, j, k):
        in_offset = c // out_group_size * in_group_size
        t['out'][n, c, h, w] = t['out'][n, c, h, w] \
                               + t['x'][n, i + in_offset, h * stride_height + j, w * stride_width + k] \
                               * t['weight'][c, i, j, k]

    res = {}
    for f in [stmt_init, stmt_calc]:
        res[f.__name__] = Statement.from_calc(f)
    return res


class PlainGroupedConv2d(PlainConv2d):
    required_args = PlainConv2d.required_args + ['groups']
    calculated_args = {**PlainConv2d.calculated_args, **{
        'in_group_size': lambda **a: a['in_channel'] // a['groups'],
        'out_group_size': lambda **a: a['out_channel'] // a['groups'],
    }}
    schedule_factory = schedule
    tensors_factory = tensors
    statements_factory = statements
    topi_cuda_task_name = 'group_conv2d_nchw.cuda'

    def topi_cuda_args(self, x=None, weight=None, out=None):
        return [x, weight, [self.stride_height, self.stride_width], 0, 1, self.groups, out.dtype]

    topi_cuda_calc_func = topi.cuda.group_conv2d_nchw
    topi_cuda_schedule_func = topi.cuda.schedule_group_conv2d_nchw
    topi_cuda_calc_ret_map = ['out']


class GroupedConv2d(Conv2d):
    def __init__(self, groups=1, **kwargs):
        super().__init__(**kwargs)
        op_idx = self._ops.index(self.conv)
        self.conv = PlainGroupedConv2d(name=self.name + '.conv', groups=groups, **self.conv.arguments)
        self.weight = self.conv.tensors['weight']
        self._ops[op_idx] = self.conv


import tvm
import numpy
from .base import calc_mode
ctx = tvm.gpu()
x = tvm.nd.array(numpy.random.random((8, 64, 24, 24)).astype('float32'), ctx=ctx)
import torch
tconv1 = torch.nn.Conv2d(64, 32, 7, 2, 3, True, groups=32)
out_t = tconv1(torch.tensor(x.asnumpy())).detach().cpu().numpy()
conv1 = GroupedConv2d(
    batch=8,
    in_channel=64, in_height=24, in_width=24,
    out_channel=32, kernel_height=7, kernel_width=7,
    stride_height=2, stride_width=2,
    pad_top=3, pad_bottom=3, pad_left=3, pad_right=3,
    biased=True, groups=32, name='conv1'
)
conv1.weight = tconv1.weight.detach().cpu().numpy()
conv1.bias = tconv1.bias.detach().cpu().numpy()
with calc_mode.under('tvm_cuda_timing'):
    conv1.imp(do_shared_opt=True, tune_kwargs={'n_trial': 8})
    out_a = conv1.calc(x)
with calc_mode.under('tvm_topi_cuda_timing'):
    conv1.imp(tune_kwargs={'n_trial': 8})
    out_b = conv1.calc(x)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy(), 0.01, 1e-3)
tvm.testing.assert_allclose(out_a.asnumpy(), out_t, 0.01, 1e-3)
tvm.testing.assert_allclose(out_b.asnumpy(), out_t, 0.01, 1e-3)
