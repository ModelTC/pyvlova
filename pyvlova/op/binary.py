import topi

from .base import ArgumentedOp
from ..poly.poly import TensorTable


from .unary import schedule


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


class ElementwiseAdd(BinaryElementwise):
    @staticmethod
    def statements_factory(**_):
        def stmt(t, n, c, h, w):
            t['out'][n, c, h, w] = t['x'][n, c, h, w] + t['y'][n, c, h, w]

        return [stmt]

    topi_cuda_calc_func = topi.add
    topi_cuda_schedule_func = topi.cuda.schedule_elemwise


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
    topi_cuda_schedule_func = topi.cuda.schedule_elemwise


'''
import tvm
import numpy
from .base import calc_mode
ctx = tvm.gpu()
x = tvm.nd.array(numpy.random.random((1, 64, 224, 224)).astype('float32'), ctx=ctx)
y = tvm.nd.array(numpy.random.random((1, 64, 224, 224)).astype('float32'), ctx=ctx)
elewise_add = PlainAdd(channel=64, height=224, width=224)
with calc_mode.under('tvm_cuda_timing'):
    elewise_add.imp(tune_kwargs={'n_trial': 1})
    out_a = elewise_add.calc(x, y)
with calc_mode.under('tvm_topi_cuda_timing'):
    elewise_add.imp(tune_kwargs={'n_trial': 1})
    out_b = elewise_add.calc(x, y)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy())
bias = tvm.nd.array(numpy.random.random((64, )).astype('float32'), ctx=ctx)
plain_bias = PlainChannelwiseAdd(channel=64, height=224, width=224)
with calc_mode.under('tvm_cuda_timing'):
    plain_bias.imp(tune_kwargs={'n_trial': 1})
    out_a = plain_bias.calc(x, bias)
with calc_mode.under('tvm_topi_cuda_timing'):
    plain_bias.imp(tune_kwargs={'n_trial': 1})
    out_b = plain_bias.calc(x, bias)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy())
'''
