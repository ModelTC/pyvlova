from pyvlova.op.base import BaseOp
from pyvlova.op.conv import Conv2d
from pyvlova.op.flatten import Flatten2d
from pyvlova.op.grouped_conv import GroupedConv2d
from pyvlova.op.pool import Pool


def shape2d(in_shape):
    if isinstance(in_shape, BaseOp):
        prev, in_shape = in_shape, []
        for i in ['batch', 'channel', 'height', 'width']:
            if hasattr(prev, i):
                in_shape.append(getattr(prev, i))
            elif hasattr(prev, 'out_' + i):
                in_shape.append(getattr(prev, 'out_' + i))
            else:
                assert False
    return in_shape


def pool(name, in_shape, kernel, stride, pad, pool_type):
    in_shape = shape2d(in_shape)
    return Pool(
        batch=in_shape[0], channel=in_shape[1], in_height=in_shape[2], in_width=in_shape[3],
        kernel_height=kernel, kernel_width=kernel,
        stride_height=stride, stride_width=stride,
        pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad,
        name=name, pool_type=pool_type
    )


def conv(name, in_shape, out_channel, kernel, stride=1, pad=0, biased=True):
    in_shape = shape2d(in_shape)
    return Conv2d(
        batch=in_shape[0], in_channel=in_shape[1], in_height=in_shape[2], in_width=in_shape[3],
        out_channel=out_channel, kernel_height=kernel, kernel_width=kernel,
        stride_height=stride, stride_width=stride,
        pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad,
        biased=biased, name=name
    )


def grouped_conv(name, in_shape, out_channel, kernel, stride=1, pad=0, groups=1, biased=True):
    in_shape = shape2d(in_shape)
    return GroupedConv2d(
        batch=in_shape[0], in_channel=in_shape[1], in_height=in_shape[2], in_width=in_shape[3],
        out_channel=out_channel, kernel_height=kernel, kernel_width=kernel,
        stride_height=stride, stride_width=stride,
        pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad,
        biased=biased, groups=groups, name=name
    )


def flatten2d(name, in_shape):
    in_shape = shape2d(in_shape)
    return Flatten2d(name=name, batch=in_shape[0], channel=in_shape[1], height=in_shape[2], width=in_shape[3])


def mock(cls, name, prev):
    in_shape = dict()
    for i in ['batch', 'channel', 'height', 'width']:
        if hasattr(prev, i):
            in_shape[i] = getattr(prev, i)
        elif hasattr(prev, 'out_' + i):
            in_shape[i] = getattr(prev, 'out_' + i)
        else:
            assert False
    return cls(name=name, **in_shape)


