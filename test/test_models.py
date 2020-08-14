import unittest

import tvm
import numpy

from pyvlova.models import *
from pyvlova import calc_mode


n_trials = 8


class TestModels(unittest.TestCase):
    def setUp(self):
        self.ctx = tvm.gpu()
    
    def _get_array(self, shape):
        return tvm.nd.array(numpy.random.random(shape).astype('float32'), ctx=self.ctx)

    def test_resnet18(self):
        model = resnet18()
        x = self._get_array((1, 3, 224, 224))
        with calc_mode.under('tvm_cuda_timing'):
            model.imp(tune_kwargs={'n_trial': n_trials})
            out_a = model.calc(x)
        with calc_mode.under('tvm_topi_cuda_timing'):
            model.imp(tune_kwargs={'n_trial': n_trials})
            out_b = model.calc(x)
        tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy(), 1e-3)

    def test_mobilenet_v2(self):
        model = MobileNetV2('mobilenet', [1, 3, 224, 224])
        x = self._get_array((1, 3, 224, 224))
        with calc_mode.under('tvm_cuda_timing'):
            model.imp(tune_kwargs={'n_trial': n_trials})
            out_a = model.calc(x)
        with calc_mode.under('tvm_topi_cuda_timing'):
            model.imp(tune_kwargs={'n_trial': n_trials})
            out_b = model.calc(x)
        tvm.testing.assert_allclose(out_a.asnumpy(), out_b.asnumpy(), 1e-3)


if __name__ == '__main__':
    unittest.main()
