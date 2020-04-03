import numpy as np
import sympy
from ..trace import current_status


class Dumpable(object):
    def dump(self, attr, target):
        v = getattr(self, attr)
        if hasattr(v, f'to_{target}'):
            return getattr(v, f'to_{target}')()
        return str(v)


class Assignment(Dumpable):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def to_loopy(self):
        left = self.dump('left', 'loopy')
        right = self.dump('right', 'loopy')
        return f'{left} = {right}'


class KthItem(Dumpable):
    def __init__(self, base, kth):
        self.base = base
        self.kth = kth

    def to_loopy(self):
        outter = self.base
        offset = self.kth
        inner_count = outter.shape[0]
        while isinstance(outter.symbol, KthItem):
            b, k = outter.symbol.base, outter.symbol.kth
            offset += k * inner_count
            inner_count *= b.shape[0]
            outter = b
        return f'{outter.to_loopy()}[{offset}]'


class DummyTensor(Dumpable):
    def __init__(self, symbol, shape=None, dtype=np.float32, wrapped=False):
        self.wrapped = wrapped
        self.symbol = symbol
        self.shape = shape or []
        self.dtype = dtype

    def to_loopy(self):
        return self.dump('symbol', 'loopy')

    def __getitem__(self, key):
        if not self.shape:
            return 
        if hasattr(key, '__iter__'):
            k, *r = key
            if r:
                return self[k][r]
            return self[k]
        if len(self.shape) == 1:
            return sympy.Symbol(KthItem(self, key).to_loopy())
        return DummyTensor(KthItem(self, key), self.shape[1:], self.dtype, wrapped=True)

    def __setitem__(self, key, value):
        left = self[key]
        assert not hasattr(left, 'shape') or not left.shape, 'there is no way to set values for arrays'
        if current_status.tracing:
            current_status.add_statement(Assignment(left, value))
