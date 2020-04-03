import os
import time
from random import getrandbits

import tvm
from tvm.autotvm.measure.measure_methods import BuildResult, gpu_verify_pass, set_cuda_target_arch, LocalBuilder
from tvm.autotvm.task.space import InstantiationError
from tvm.contrib import tar, ndk
from tvm.target import build_config


def _poly_build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    target, task, config = measure_input
    with target:
        lowered_func, tensors = task.instantiate(config)
        if not config.valid:
            raise InstantiationError(config.errors)
        opts = build_option or {}
        if check_gpu:
            opts["add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)
        with build_config(**opts):
            func = tvm.build(lowered_func)
    return func, [(i.shape, i.dtype) for i in tensors]


def _poly_wrap_build_func(build_func):
    if not hasattr(build_func, "output_format"):
        raise AttributeError("Expect build_func to have the attribute output_format.")
    output_format = build_func.output_format

    def _wrapped(measure_input, tmp_dir, **kwargs):
        tic = time.time()
        try:
            filename = os.path.join(tmp_dir, "tmp_func_%0x.%s" % (
                getrandbits(64), output_format))
            func, arg_info = _poly_build_func_common(measure_input, **kwargs)
            func.export_library(filename, build_func)
        except Exception as e:
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)
    return _wrapped


class PolyLocalBuilder(LocalBuilder):
    def __init__(self, build_func='default', **kwargs):
        super().__init__(**kwargs)
        if isinstance(build_func, str):
            if build_func == 'default':
                build_func = tar.tar
            elif build_func == 'ndk':
                build_func = ndk.create_shared
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _poly_wrap_build_func(build_func)
