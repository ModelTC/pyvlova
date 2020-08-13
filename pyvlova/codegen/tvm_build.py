import tvm
from tvm.te import schedule
from tvm.driver import build_module


def build_tvm_stmt(stmt, args, binds=None, name='main',
        target=None, target_host=None):
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
    module = build_module.lower( module, args, name=name)

    kernel = build_module.build(
        module, name=name, target=target, target_host=target_host)

    return kernel
