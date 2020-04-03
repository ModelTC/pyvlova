from .base import ArgumentedOp
from ..poly.poly import TensorTable, Statement
from ..poly.schedule_tree import ScheduleTree


def schedule(**kwargs):
    copy_t = 'stmt_copy[n, c, h, w]'
    zero_t = 'stmt_zero[n, c, h, w]'
    output_constraints = '0 <= n < batch and 0 <= c < channel ' \
                         'and 0 <= h < out_height and 0 <= w < out_width'
    copy_constraints = '0 <= n < batch and 0 <= c < channel ' \
                       'and 0 <= h - pad_top < in_height and 0 <= w - pad_left < in_width'
    domain = '[batch, channel, in_height, in_width, out_height, out_width, ' \
             'pad_top, pad_left] -> {' \
             f'{copy_t}: {copy_constraints}; ' \
             f'{zero_t}: {output_constraints} and not ({copy_constraints})' \
             '}'
    outer_schedule = '[%s]' % ', '.join(map(
        lambda x: f'{{{copy_t}->[({x})];{zero_t}->[({x})]}}', ('n', 'c', 'h', 'w')))

    tree = ScheduleTree.from_yaml(f'''
    domain: "{domain}"
    child:
        schedule: "{outer_schedule}"
        permutable: 1
        coincident: [1, 1, 1, 1]
        child:
            set:
              - filter: "{{{zero_t}}}"
              - filter: "{{{copy_t}}}"
    ''')
    tree.apply_params(**kwargs)
    return tree


def tensors(batch=1, channel=1, in_height=1, in_width=1, out_height=1, out_width=1, **_):
    table = TensorTable()
    table.add_tensor('x', [batch, channel, in_height, in_width])
    table.add_tensor('out', [batch, channel, out_height, out_width])
    return table


def statements(pad_top=0, pad_left=0, **_):
    def stmt_zero(t, n, c, h, w):
        t['out'][n, c, h, w] = 0.0

    def stmt_copy(t, n, c, h, w):
        t['out'][n, c, h, w] = t['x'][n, c, h - pad_top, w - pad_left]

    res = {}
    for i in [stmt_zero, stmt_copy]:
        res[i.__name__] = Statement.from_calc(i)
    return res


class PlainPadding(ArgumentedOp):
    required_params = [
        'channel', 'in_height', 'in_width',
        'pad_top', 'pad_bottom', 'pad_left', 'pad_right',
    ]
    optional_params = {
        'batch': 1,
    }
    calculated_params = {
        'out_height': lambda **a: a['in_height'] + a['pad_top'] + a['pad_bottom'],
        'out_width': lambda **a: a['in_width'] + a['pad_left'] + a['pad_right'],
    }
    tensor_order = ['x', 'out']
    inputs = ['x']
    outputs = ['out']
    schedule_factory = schedule
    tensors_factory = tensors
    statements_factory = statements
