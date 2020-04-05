import topi
from tvm import te

from .base import ArgumentedOp, SequenceOp
from .padding import Padding
from ..poly.poly import TensorTable, Statement, trace_mode
from ..poly.schedule_tree import ScheduleTree
from ..utils import tir_imm


def schedule(**kwargs):
    init_t = 'stmt_init[n, c, h, w]'
    calc_t = 'stmt_calc[n, c, h, w, i, j]'
    output_constraints = '0 <= n < batch and 0 <= c < channel ' \
                         'and 0 <= h < out_height and 0 <= w < out_width'
    calc_constraints = '0 <= i < kernel_height and 0 <= j < kernel_width'
    domain = '[batch, channel, in_height, in_width, out_height, out_width, ' \
             'kernel_height, kernel_width] -> {' \
             f'{init_t}: {output_constraints}; ' \
             f'{calc_t}: {output_constraints} and {calc_constraints}' \
             '}'
    outer_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{init_t}->[({x})];{calc_t}->[({x})]}}', ('n', 'c', 'h', 'w')))
    inner_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{calc_t}->[({x})]}}', ('i', 'j')))

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
                    coincident: [1, 1]
    ''')
    tree.apply_params(**dict(filter(lambda x: isinstance(x[1], int), kwargs.items())))
    return tree


def tensors(batch=1, channel=1, in_height=1, in_width=1, out_height=1, out_width=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, channel, in_height, in_width])
    table.add_tensor('out', [batch, channel, out_height, out_width])
    return table


def statements(stride_height=1, stride_width=1, kernel_height=1, kernel_width=1, pool_type='max', **_):
    def stmt_init(t, n, c, h, w):
        if pool_type == 'max':
            t['out'][n, c, h, w] = t['x'][n, c, h * stride_height, w * stride_height]
        else:
            t['out'][n, c, h, w] = 0.0

    def stmt_calc(t, n, c, h, w, i, j):
        if trace_mode.mode == 'tvm':
            if pool_type == 'max':
                t['out'][n, c, h, w] = te.max(
                    t['out'][n, c, h, w], t['x'][n, c, h * stride_height + i, w * stride_width + j])
            else:
                t['out'][n, c, h, w] = t['out'][n, c, h, w] \
                                       + t['x'][n, c, h * stride_height + i, w * stride_width + j] \
                                       / tir_imm(float(kernel_height * kernel_width))
        elif trace_mode.mode == 'tensor_access':
            t['out'][n, c, h, w] = t['x'][n, c, h, w]
        else:
            if pool_type == 'max':
                t['out'][n, c, h, w] = max(t['x'][n, c, h, w],
                                           t['x'][n, c, h * stride_height + i, w * stride_width + j])
            else:
                t['out'][n, c, h, w] = t['out'][n, c, h, w] \
                                       + t['x'][n, c, h * stride_height + i, w * stride_width + j] \
                                       / float(kernel_height * kernel_width)

    res = {}
    for f in [stmt_init, stmt_calc]:
        res[f.__name__] = Statement.from_calc(f)
    return res


class PlainPool(ArgumentedOp):
    required_args = [
        'channel', 'in_height', 'in_width',
        'kernel_height', 'kernel_width', 'pool_type',
    ]
    optional_args = {
        'batch': 1, 'stride_height': 1, 'stride_width': 1
    }
    calculated_args = {
        'out_height': lambda **a: (a['in_height'] - a['kernel_height']) // a['stride_height'] + 1,
        'out_width': lambda **a: (a['in_width'] - a['kernel_width']) // a['stride_width'] + 1,
    }
    tensor_order = ['x', 'out']
    inputs = ['x']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    statements_factory = statements

    def topi_cuda_args(self, x=None, out=None):
        return [x, [self.kernel_height, self.kernel_width],
                [self.stride_height, self.stride_width],
                [0, 0, 0, 0], self.pool_type]

    topi_cuda_calc_func = topi.nn.pool
    topi_cuda_schedule_func = lambda outs: topi.cuda.schedule_pool(outs, 'NCHW')
    topi_cuda_calc_ret_map = ['out']


class Pool(SequenceOp):
    def __init__(self, batch=1, channel=1, in_height=1, in_width=1,
                 kernel_height=1, kernel_width=1, stride_height=1, stride_width=1,
                 pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, pool_type='max', name=''):
        super().__init__(name=name)
        if pad_top or pad_bottom or pad_left or pad_right:
            self.pad = Padding(
                name=self.name + '.pad', batch=batch,
                channel=channel, in_height=in_height, in_width=in_width,
                pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right
            )
            self._ops.append(self.pad)
            in_height = self.pad.out_height
            in_width = self.pad.out_width
        else:
            self.pad = None
        self.pool = PlainPool(
            name=self.name + '.conv', batch=batch,
            channel=channel, in_height=in_height, in_width=in_width,
            kernel_height=kernel_height, kernel_width=kernel_width,
            stride_height=stride_height, stride_width=stride_width,
            pool_type=pool_type
        )
        self._ops.append(self.pool)

        for i in ['batch', 'channel', 'out_height', 'out_width']:
            setattr(self, i, getattr(self.pool, i))


class AdaptivePool(PlainPool):
    required_args = [
        'channel', 'in_height', 'in_width',
        'out_height', 'out_width', 'pool_type',
    ]
    optional_args = {
        'batch': 1,
    }
    calculated_args = {
        'stride_height': lambda **a: a['in_height'] // a['out_height'],
        'kernel_height': lambda **a: a['in_height'] - (a['out_height'] - 1) * a['stride_height'],
        'stride_width': lambda **a: a['in_width'] // a['out_width'],
        'kernel_width': lambda **a: a['in_width'] - (a['out_width'] - 1) * a['stride_width'],
    }

    def topi_cuda_args(self, x=None, out=None):
        return [x, [self.out_height, self.out_width], self.pool_type]

    topi_cuda_calc_func = topi.nn.adaptive_pool
    topi_cuda_schedule_func = lambda outs: topi.cuda.schedule_adaptive_pool(outs)
    topi_cuda_calc_ret_map = ['out']


'''
import tvm
import numpy
from .base import calc_mode
ctx = tvm.gpu()
x = tvm.nd.array(numpy.random.random((1, 64, 224, 224)).astype('float32'), ctx=ctx)
maxpool = PlainPool(channel=64, in_height=224, in_width=224,
                        kernel_height=7, kernel_width=7, stride_height=2, stride_width=2, pool_type='max')
with calc_mode.under('tvm_cuda_timing'):
    maxpool.imp(tune_kwargs={'n_trial': 1})
    out_a = maxpool.calc(x)
with calc_mode.under('tvm_topi_cuda_timing'):
    maxpool.imp(tune_kwargs={'n_trial': 1})
    out_b = maxpool.calc(x)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy())
avgpool = PlainPool(channel=64, in_height=224, in_width=224,
                    kernel_height=224, kernel_width=224, pool_type='avg')
with calc_mode.under('tvm_cuda_timing'):
    avgpool.imp(tune_kwargs={'n_trial': 1})
    out_a = avgpool.calc(x)
with calc_mode.under('tvm_topi_cuda_timing'):
    avgpool.imp(tune_kwargs={'n_trial': 1})
    out_b = avgpool.calc(x)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy(), 1e-3)
adpavgpool = PlainAdaptivePool(channel=64, in_height=224, in_width=224,
                               out_height=1, out_width=1, pool_type='avg')
with calc_mode.under('tvm_cuda_timing'):
    adpavgpool.imp(tune_kwargs={'n_trial': 1})
    out_a = adpavgpool.calc(x)
with calc_mode.under('tvm_topi_cuda_timing'):
    adpavgpool.imp(tune_kwargs={'n_trial': 1})
    out_b = adpavgpool.calc(x)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy(), 1e-3)
'''
