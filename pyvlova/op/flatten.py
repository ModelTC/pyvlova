import topi

from ..poly.poly import TensorTable
from .unary import UnaryElementwise


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


'''
import tvm
import numpy
from .base import calc_mode
ctx = tvm.gpu()
x = tvm.nd.array(numpy.random.random((1, 64, 224, 224)).astype('float32'), ctx=ctx)
flatten2d = Flatten2d(channel=64, height=224, width=224)
with calc_mode.under('tvm_cuda_timing'):
    flatten2d.imp(tune_kwargs={'n_trial': 1})
    out_a = flatten2d.calc(x)
with calc_mode.under('tvm_topi_cuda_timing'):
    flatten2d.imp(tune_kwargs={'n_trial': 1})
    out_b = flatten2d.calc(x)
tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy())
'''
