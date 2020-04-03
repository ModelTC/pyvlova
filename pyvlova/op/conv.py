from .base import ArgumentedOp
from ..poly.poly import TensorTable, Statement
from ..poly.schedule_tree import ScheduleTree


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
    for i in [stmt_init, stmt_calc]:
        res[i.__name__] = Statement.from_calc(i)
    return res


class PlainConv2d(ArgumentedOp):
    required_params = [
        'in_channel', 'in_height', 'in_width', 'out_channel',
        'kernel_height', 'kernel_width',
    ]
    optional_params = {
        'batch': 1, 'stride_height': 1, 'stride_width': 1
    }
    calculated_params = {
        'out_height': lambda **a: (a['in_height'] - a['kernel_height']) // a['stride_height'] + 1,
        'out_width': lambda **a: (a['in_width'] - a['kernel_width']) // a['stride_width'] + 1,
    }
    tensor_order = ['x', 'weight', 'out']
    inputs = ['x', 'weight']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    statements_factory = statements
