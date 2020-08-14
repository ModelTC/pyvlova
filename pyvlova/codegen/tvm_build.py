# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0
import tvm
from tvm.te import schedule
from tvm.driver import build_module

from ..utils import slugify


def lower_tvm_stmt(stmt, args, binds=None, name='main'):
    name = slugify(name)
    compact = schedule.VerifyCompactBuffer(stmt)
    binds, arg_list = build_module.get_binds(
        args, compact=compact, binds=binds)

    func = schedule.SchedulePostProcToPrimFunc(
        arg_list, stmt, binds)

    func = func.with_attr('global_symbol', name)

    pass_ctx = tvm.ir.transform.PassContext.current()
    if pass_ctx.config.get('tir.noalias', True):
        func = func.with_attr('tir.noalias', True)

    module = tvm.IRModule({name: func})
    module = build_module.lower(module, args, name=name)

    return module
