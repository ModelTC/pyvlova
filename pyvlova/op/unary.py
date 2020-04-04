from tvm import te

from .base import ArgumentedOp
from ..poly.poly import TensorTable, Statement, trace_mode
from ..poly.schedule_tree import ScheduleTree
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
    required_params = [
        'channel', 'height', 'width',
    ]
    optional_params = {
        'batch': 1,
    }
    calculated_params = {}
    tensor_order = ['x', 'out']
    inputs = ['x']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors


class PlainReLU(UnaryElementwise):
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
